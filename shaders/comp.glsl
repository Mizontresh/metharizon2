#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(binding=0, rgba8) uniform writeonly image2D img;
layout(binding=1) uniform Camera {
    vec3 pos;
    vec3 forward;
    vec3 up;
    vec3 right;
} cam;

layout(binding=2) uniform Objects {
    vec4 obj[2]; // xyz = position, w = radius
} objs;

bool raySphere(vec3 ro, vec3 rd, vec3 center, float radius, out float tHit) {
    vec3 oc = ro - center;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - radius*radius;
    float h = b*b - c;
    if(h < 0.0) return false;
    h = sqrt(h);
    float t0 = -b - h;
    float t1 = -b + h;
    tHit = t0 > 0.0 ? t0 : t1;
    return tHit > 0.0;
}

// Quaternion Julia distance estimator (bounded ~radius 4)
float quatJuliaDE(vec3 p) {
    vec4 z = vec4(0.0, p);
    const vec4 c = vec4(-0.2, 0.7, 0.0, 0.0);
    float dr = 1.0;
    const int ITER = 12;
    const float power = 8.0;
    for(int i=0;i<ITER;i++){
        float r = length(z);
        if(r>4.0) break;
        float theta = acos(z.w/r);
        float phi   = atan(length(z.yzw), z.x);
        float psi   = atan(z.y, z.z);
        dr = pow(r, power-1.0)*power*dr + 1.0;
        float rp = pow(r, power);
        theta *= power; phi *= power; psi *= power;
        z = rp * vec4(cos(theta),
                      sin(theta)*sin(phi)*cos(psi),
                      sin(theta)*sin(phi)*sin(psi),
                      sin(theta)*cos(phi)) + c;
    }
    return length(z)/dr;
}

float sceneDE(vec3 p){
    float d0 = quatJuliaDE(p - objs.obj[0].xyz);
    float d1 = quatJuliaDE(p - objs.obj[1].xyz);
    return min(d0, d1);
}

vec3 sceneNormal(vec3 p){
    float e = 0.0005;
    return normalize(vec3(
        sceneDE(p+vec3(e,0,0)) - sceneDE(p-vec3(e,0,0)),
        sceneDE(p+vec3(0,e,0)) - sceneDE(p-vec3(0,e,0)),
        sceneDE(p+vec3(0,0,e)) - sceneDE(p-vec3(0,0,e))
    ));
}

// estimate normal of the scene
vec3 getNormal(vec3 p) {
    float e = 0.0005;
    return normalize(vec3(
        sceneDE(p+vec3(e,0,0)) - sceneDE(p-vec3(e,0,0)),
        sceneDE(p+vec3(0,e,0)) - sceneDE(p-vec3(0,e,0)),
        sceneDE(p+vec3(0,0,e)) - sceneDE(p-vec3(0,0,e))
    ));
}

void main(){
    ivec2 uv = ivec2(gl_GlobalInvocationID.xy);
    if(uv.x >= imageSize(img).x || uv.y >= imageSize(img).y) return;

    // generate ray
    vec2 frag = (vec2(uv) / vec2(imageSize(img)) - 0.5) * 2.0;
    frag.x *= float(imageSize(img).x)/imageSize(img).y;
    vec3 rd = normalize(frag.x*cam.right + frag.y*cam.up + cam.forward);
    vec3 ro = cam.pos;

    float tSphere = 1e9;
    float tmp;
    if(raySphere(ro, rd, objs.obj[0].xyz, objs.obj[0].w, tmp) && tmp < tSphere) tSphere = tmp;
    if(raySphere(ro, rd, objs.obj[1].xyz, objs.obj[1].w, tmp) && tmp < tSphere) tSphere = tmp;


    // ray march
    float t = 0.0;
    const float MAXT = 50.0;
    float d;
    for(int i=0;i<128;i++){
        vec3 p = ro + rd*t;
        d = sceneDE(p);
        if(d < 0.001 || t > MAXT) break;
        t += d;
    }

    vec3 col;
    if(t > MAXT) {
        col = vec3(0.0);
    } else {
        vec3 p = ro + rd*t;
        vec3 n = getNormal(p);
        // simple side lighting
        vec3 lightDir = normalize(vec3(1.0, 1.0, 0.5));
        float diff = clamp(dot(n, lightDir), 0.0, 1.0);
        float da = quatJuliaDE(p - objs.obj[0].xyz);
        float db = quatJuliaDE(p - objs.obj[1].xyz);
        vec3 baseCol = da < db ? vec3(0.6,0.8,1.0) : vec3(1.0,0.6,0.4);
        col = mix(vec3(0.1,0.1,0.2), baseCol, diff);
    }

    if(tSphere < min(t, MAXT))
        col = mix(col, vec3(1.0,1.0,0.0), 0.3);

    imageStore(img, uv, vec4(col,1.0));
}
