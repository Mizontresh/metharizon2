#version 450

layout(local_size_x = 16, local_size_y = 16) in;

struct Body {
    vec3 pos;    float mass;
    vec3 vel;    float pad0;
    vec3 angVel; float pad1;
    vec4 orient; float maxIter;
    vec3 pad2;
};

layout(std430, binding = 0) buffer Bodies {
    Body bodies[];
};

layout(std140, binding = 1) uniform Camera {
    vec3 camPos;
    vec3 camForward;
    vec3 camUp;
    vec3 camRight;
};

layout(rgba8, binding = 2) writeonly uniform image2D img;

const int MAX_STEPS = 128;
const float THRESH = 4.0;

mat3 quatToMat(vec4 q) {
    float x=q.x,y=q.y,z=q.z,w=q.w;
    return mat3(
      1-2*(y*y+z*z),   2*(x*y - w*z), 2*(x*z + w*y),
      2*(x*y + w*z), 1-2*(x*x+z*z),   2*(y*z - w*x),
      2*(x*z - w*y),   2*(y*z + w*x), 1-2*(x*x+y*y)
    );
}

void main(){
    ivec2 uv = ivec2(gl_GlobalInvocationID.xy);
    ivec2 res = imageSize(img);
    if (uv.x>=res.x||uv.y>=res.y) return;

    vec2 xy = (vec2(uv)/vec2(res)-0.5)*2.0;
    vec3 rayDir = normalize(camForward + xy.x*camRight + xy.y*camUp);

    vec4 color = vec4(0.0);
    // loop over bodies
    for (int i = 0; i < bodies.length(); ++i) {
        Body b = bodies[i];
        // transform ray into body‐local space
        mat3 rot = quatToMat(b.orient);
        vec3 localOrigin = rot * (camPos - b.pos);
        vec3 localDir    = rot * rayDir;

        // basic fractal: raymarch Mandelbulb
        vec3 z = localOrigin;
        float dr = 1.0;
        float r = 0.0;
        int iter = 0;
        for (int s = 0; s < MAX_STEPS; ++s){
            r = length(z);
            if (r>THRESH) break;
            // Mandelbulb power 8
            float theta = acos(z.z/r);
            float phi   = atan(z.y,z.x);
            float zr    = pow(r,8.0);
            theta *= 8.0;
            phi   *= 8.0;
            z = zr*vec3(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)) + localOrigin;
            dr = pow(r,7.0)*8.0*dr + 1.0;
            iter = s;
        }

        // shade by escape time, mass, velocity magnitude
        float shade = iter==MAX_STEPS ? 0.0 : float(iter)/float(MAX_STEPS);
        float vmag  = length(b.vel);
        vec3 col = mix(vec3(0.2,0.4,0.6), vec3(1.0,0.8,0.2), shade) * (1.0+vmag*0.1);
        color.rgb += col * (1.0/float(bodies.length()));
    }

    imageStore(img, uv, vec4(color.rgb,1.0));
}
