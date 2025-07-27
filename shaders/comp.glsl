#version 450

// tweak these to taste
layout(local_size_x = 16, local_size_y = 16) in;
layout(binding = 0, rgba8) uniform writeonly image2D img;
const int   MAX_ITERS   = 20;
const float BOX_SCALE   = 2.2;
const float ESCAPE      = 10.0;

// box‐fold into [-1,1]
vec3 boxFold(vec3 p){
    return clamp(p, -1.0, 1.0) * 2.0 - p;
}
// sphere‐fold (invert/expand)
vec3 sphereFold(vec3 p){
    float r2 = dot(p,p);
    if (r2 < 0.25)      p *= 4.0;
    else if (r2 < 1.0)  p /= r2;
    return p;
}
// Mandelbox distance estimator
float mandelboxDE(vec3 pos){
    vec3  z  = pos;
    float dr = 1.0;
    for (int i = 0; i < MAX_ITERS; i++){
        z  = boxFold(z);
        z  = sphereFold(z);
        z  = z * BOX_SCALE + pos;
        dr = dr * abs(BOX_SCALE) + 1.0;
        if (length(z) > ESCAPE) break;
    }
    return length(z) / dr;
}

// simple HSV→RGB palette
vec3 palette(float t){
    // hue cycles with t
    float h = mod(0.7 + t*1.5, 1.0);
    float s = 0.8;
    float v = t;
    // approximate HSV→RGB
    vec3 k = vec3(1.0, 2.0/3.0, 1.0/3.0);
    vec3 p = abs(fract(vec3(h) + k) * 6.0 - 3.0) - 1.0;
    return clamp(mix(k.xxx, p, vec3(s)), 0.0, 1.0) * v;
}

void main(){
    ivec2  px = ivec2(gl_GlobalInvocationID.xy);
    ivec2  sz = imageSize(img);
    vec2   uv = (vec2(px) / vec2(sz)) * 2.0 - 1.0;
    uv.x *= float(sz.x) / float(sz.y);

    // camera
    vec3 ro = vec3(0.0, 0.0, -2.5);
    vec3 rd = normalize(vec3(uv, 1.5));

    // ray‐march
    float t = 0.0;
    int steps = 0;
    for (; steps < 128; ++steps){
        vec3  p = ro + rd * t;
        float d = mandelboxDE(p);
        if (d < 0.0005) break;
        t += d;
    }

    // color by how quickly it converged
    float f = float(steps) / 128.0;
    vec3  col = palette(1.0 - f);

    imageStore(img, px, vec4(col,1.0));
}
