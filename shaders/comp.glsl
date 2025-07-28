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


// Sierpinski tetrahedron distance estimator
float sierpinski(vec3 p){
    const float SCALE = 2.0;
    const float OFFSET = 1.0;
    float m = 1.0;
    for(int i=0;i<6;i++){
        p = abs(p);
        if(p.x < p.y) p.xy = p.yx;
        if(p.x < p.z) p.xz = p.zx;
        if(p.y < p.z) p.yz = p.zy;
        p = SCALE * p - (SCALE - 1.0) * OFFSET;
        m *= SCALE;
main
    }
    return length(p)/m - 0.1;
}

float objectDE(int idx, vec3 p){
    vec3 lp = quatRotateInv(objs.quat[idx], p - objs.posRad[idx].xyz);
    float r = objs.posRad[idx].w;

    return sierpinski(lp / r) * r;
main
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
