#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0, rgba8) uniform writeonly image2D img;
layout(binding = 1) uniform Camera {
    vec3 camPos;
    vec3 forward;
    vec3 up;
    vec3 right;
} cam;

const int MAX_STEPS = 100;
const float MAX_DIST = 100.0;
const float MIN_DIST = 0.001;

// distance estimator for a simple Mandelbulb fold fractal
float mandelbulbDE(vec3 p) {
    vec3 z = p;
    float dr = 1.0;
    float r = 0.0;
    for(int i = 0; i < 8; i++) {
        r = length(z);
        if (r > 2.0) break;
        // convert to polar
        float theta = acos(z.z/r);
        float phi = atan(z.y, z.x);
        float power = 8.0;
        // scale & rotate the point
        float zr = pow(r, power);
        theta *= power;
        phi *= power;
        // compute new position
        z = zr * vec3(sin(theta)*cos(phi), sin(phi)*sin(theta), cos(theta));
        z += p;
        dr =  pow(r, power-1.0)*power*dr + 1.0;
    }
    return 0.5*log(r)*r/dr;
}

vec3 rayDirection(vec2 uv) {
    // uv in [-1,1]
    return normalize(uv.x*cam.right + uv.y*cam.up + cam.forward);
}

vec3 rayMarch(vec3 ro, vec3 rd) {
    float dist = 0.0;
    for(int i=0; i<MAX_STEPS; i++) {
        vec3 p = ro + rd*dist;
        float d = mandelbulbDE(p);
        if (d < MIN_DIST) break;
        dist += d;
        if (dist > MAX_DIST) break;
    }
    if (dist > MAX_DIST) return vec3(0.0);
    // simple coloring by steps
    float t = float(dist/MAX_DIST);
    return mix(vec3(0.2,0.3,0.5), vec3(1.0,0.7,0.3), t);
}

void main(){
    ivec2 pix = ivec2(gl_GlobalInvocationID.xy);
    ivec2 res = imageSize(img);
    if (pix.x >= res.x || pix.y >= res.y) return;
    vec2 uv = (vec2(pix) / vec2(res) - 0.5) * vec2(res.x/res.y,1.0) * 2.0;
    vec3 rd = rayDirection(uv);
    vec3 col = rayMarch(cam.camPos, rd);
    imageStore(img, pix, vec4(col,1.0));
}
