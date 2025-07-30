#version 450

layout(local_size_x = 64) in;

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

const float G = 6.674e-1;
const float dt = 0.016;

uint idx = gl_GlobalInvocationID.x;
uint N   = uint(bodies.length());

void main() {
    if (idx >= N) return;
    Body me = bodies[idx];

    // 1) compute net force
    vec3 force = vec3(0.0);
    for (uint j = 0; j < N; ++j) {
        if (j == idx) continue;
        Body other = bodies[j];
        vec3 d = other.pos - me.pos;
        float r2 = dot(d,d) + 1e-3;
        float invR3 = inversesqrt(r2*r2*r2);
        // gravity
        force += G * me.mass * other.mass * d * invR3;
        // repulsion based on other's maxIter
        float rep = other.maxIter > 0.0 ? 1.0 / other.maxIter : 0.0;
        force -= rep * d * invR3;
    }

    // 2) linear integration
    vec3 accel = force / me.mass;
    me.vel += accel * dt;
    me.pos += me.vel * dt;

    // 3) angular integration (quaternion derivative)
    vec4 q = me.orient;
    vec3 w = me.angVel;
    vec4 qDot = 0.5 * vec4(
        -w.x*q.x - w.y*q.y - w.z*q.z,
         w.x*q.w + w.y*q.z - w.z*q.y,
        -w.x*q.z + w.y*q.w + w.z*q.x,
         w.x*q.y - w.y*q.x + w.z*q.w
    );
    q += qDot * dt;
    bodies[idx].orient = normalize(q);

    // 4) write back
    bodies[idx].vel   = me.vel;
    bodies[idx].pos   = me.pos;
    bodies[idx].angVel= me.angVel;
    bodies[idx].orient= me.orient;
}
