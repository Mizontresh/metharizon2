#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(binding=0, rgba8) uniform writeonly image2D img;
layout(binding=1) uniform Camera {
    vec3 pos;
    vec3 forward;
    vec3 up;
    vec3 right;
} cam;

layout(binding=2, std140) uniform Objects {
    vec4 posRad[2]; // xyz = position, w = radius
    vec4 quat[2];   // quaternion (x, y, z, w)
} objs;

vec3 quatRotate(vec4 q, vec3 v){
    vec3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

vec3 quatRotateInv(vec4 q, vec3 v){
    return quatRotate(vec4(-q.xyz, q.w), v);
}

// rotate-fold Mandelbulb-ish fractal
float mandelbulb(vec3 p) {
    vec3 z = p;
    float dr = 1.0;
    float r  = 0.0;
    const int ITER = 8;
    for(int i=0;i<ITER;i++){
        r = length(z);
        if(r>2.0) break;
        // convert to polar
        float theta = acos(z.z/r);
        float phi   = atan(z.y, z.x);
        dr =  pow(r,7.0)*8.0*dr + 1.0;
        float zr = pow(r,8.0);
        theta = theta*8.0;
        phi   = phi*8.0;
        z = zr * vec3(sin(theta)*cos(phi), sin(phi)*sin(theta), cos(theta)) + p;
    }
    return 0.5*log(r)*r/dr;
}

float objectDE(int idx, vec3 p){
    vec3 lp = quatRotateInv(objs.quat[idx], p - objs.posRad[idx].xyz);
    float r = objs.posRad[idx].w;
    return mandelbulb(lp / r) * r;
}

float sceneDE(vec3 p){
    float d0 = objectDE(0, p);
    float d1 = objectDE(1, p);
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


void main(){
    ivec2 uv = ivec2(gl_GlobalInvocationID.xy);
    if(uv.x >= imageSize(img).x || uv.y >= imageSize(img).y) return;

    // generate ray
    vec2 frag = (vec2(uv) / vec2(imageSize(img)) - 0.5) * 2.0;
    frag.x *= float(imageSize(img).x)/imageSize(img).y;
    vec3 rd = normalize(frag.x*cam.right + frag.y*cam.up + cam.forward);
    vec3 ro = cam.pos;

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
        vec3 n = sceneNormal(p);
        // simple side lighting
        vec3 lightDir = normalize(vec3(1.0, 1.0, 0.5));
        float diff = clamp(dot(n, lightDir), 0.0, 1.0);
        col = mix(vec3(0.1,0.1,0.2), vec3(0.6,0.8,1.0), diff);
    }

    imageStore(img, uv, vec4(col,1.0));
}
