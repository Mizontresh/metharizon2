#pragma once
#include <cmath>
#include <algorithm>

struct Vec3 {
    float x, y, z;
};

struct Quat {
    float w, x, y, z;
};


inline Vec3 operator+(Vec3 a, Vec3 b){ return {a.x+b.x, a.y+b.y, a.z+b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b){ return {a.x-b.x, a.y-b.y, a.z-b.z}; }
inline Vec3 operator*(Vec3 a, float s){ return {a.x*s, a.y*s, a.z*s}; }
inline Vec3 operator/(Vec3 a, float s){ return {a.x/s, a.y/s, a.z/s}; }
inline Vec3 operator*(float s, Vec3 a){ return {a.x*s, a.y*s, a.z*s}; }
inline float dot(Vec3 a, Vec3 b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
inline float length(Vec3 a){ return std::sqrt(dot(a,a)); }
inline Vec3 normalize(Vec3 a){ float l = length(a); return (l>0)? a/l : Vec3{0,0,0}; }
inline Vec3 cross(Vec3 a, Vec3 b){
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}

inline Quat quatMul(const Quat& a, const Quat& b){
    return {a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
            a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
            a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
            a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w};
}

inline Quat quatFromAxisAngle(Vec3 axis, float angle){
    float half = angle*0.5f;
    float s = std::sin(half);
    axis = normalize(axis);
    return {std::cos(half), axis.x*s, axis.y*s, axis.z*s};
}

inline Quat quatFromAxisAngle(const float axis[3], float angle){
    return quatFromAxisAngle(Vec3{axis[0],axis[1],axis[2]}, angle);
}

inline void quatNormalize(Quat& q){
    float len = std::sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
    q.w/=len; q.x/=len; q.y/=len; q.z/=len;
}

inline Vec3 rotateVec(const Quat& q, Vec3 v){
    Quat p{0.f, v.x, v.y, v.z};
    Quat iq{q.w,-q.x,-q.y,-q.z};
    Quat r = quatMul(quatMul(q,p),iq);
    return {r.x,r.y,r.z};
}

inline Quat quatConjugate(const Quat& q){
    return {q.w,-q.x,-q.y,-q.z};
}

inline Vec3 rotateInv(const Quat& q, Vec3 v){
    return rotateVec(quatConjugate(q), v);
}

inline void rotateVec(const Quat& q, const float in[3], float out[3]){
    Vec3 v{in[0],in[1],in[2]};
    Vec3 r = rotateVec(q, v);
    out[0]=r.x; out[1]=r.y; out[2]=r.z;
}

typedef float (*DEFunc)(Vec3);

struct FractalObject {
    Vec3 position;
    Vec3 velocity;
    Vec3 angularVelocity;
    Quat orientation;
    float radius;
    float mass;
    float inertia;
    DEFunc de;
};

inline float sierpinskiDE(Vec3 p){
    const float SCALE = 2.0f;
    const float OFFSET = 1.0f;
    float m = 1.0f;
    for(int i=0;i<6;i++){
        p.x = std::fabs(p.x); p.y = std::fabs(p.y); p.z = std::fabs(p.z);
        if(p.x < p.y) std::swap(p.x, p.y);
        if(p.x < p.z) std::swap(p.x, p.z);
        if(p.y < p.z) std::swap(p.y, p.z);
        p.x = SCALE*p.x - (SCALE - 1.0f)*OFFSET;
        p.y = SCALE*p.y - (SCALE - 1.0f)*OFFSET;
        p.z = SCALE*p.z - (SCALE - 1.0f)*OFFSET;
        m *= SCALE;
    }
    return length(p)/m;
}

// The DE never becomes negative along the x axis so the bisection search above
// converged towards zero, resulting in a near‑zero radius. The Sierpiński
// tetrahedron comfortably fits in a unit sphere, so just return 1.
inline float estimateSierpinskiRadius(){
    return 1.0f;
}

inline void integrateOrientation(FractalObject& obj, float dt){
    Quat wq{0.f, obj.angularVelocity.x, obj.angularVelocity.y, obj.angularVelocity.z};
    Quat dq = quatMul(wq, obj.orientation);
    obj.orientation.w += 0.5f * dq.w * dt;
    obj.orientation.x += 0.5f * dq.x * dt;
    obj.orientation.y += 0.5f * dq.y * dt;
    obj.orientation.z += 0.5f * dq.z * dt;
    quatNormalize(obj.orientation);
}

inline Vec3 findContactPoint(const FractalObject& a, const FractalObject& b){
    Vec3 dir = normalize(b.position - a.position);
    float dist = length(b.position - a.position);
    float t = a.radius;
    float tMax = dist - b.radius;
    for(int i=0;i<16 && t < tMax; ++i){
        Vec3 p = a.position + dir * t;
        Vec3 la = rotateInv(a.orientation, p - a.position);
        Vec3 lb = rotateInv(b.orientation, p - b.position);
        float da = a.de ? a.de(la / a.radius) * a.radius : 0.f;
        float db = b.de ? b.de(lb / b.radius) * b.radius : 0.f;
        float d = da + db;
        if(d < 0.0005f) break;
        t += d * 0.5f;
    }
    return a.position + dir * t;
}

inline void stepPhysics(FractalObject& a, FractalObject& b, float dt, float G=1.0f){
    // gravity
    Vec3 diff = b.position - a.position;
    float distSq = dot(diff,diff) + 1e-6f;
    float dist = std::sqrt(distSq);
    Vec3 n = diff / dist;
    float forceMag = G * a.mass * b.mass / distSq;
    Vec3 force = n * forceMag;
    a.velocity = a.velocity + force / a.mass * dt;
    b.velocity = b.velocity - force / b.mass * dt;

    // integrate motion
    a.position = a.position + a.velocity * dt;
    b.position = b.position + b.velocity * dt;
    integrateOrientation(a, dt);
    integrateOrientation(b, dt);

    // bounding sphere test
    diff = b.position - a.position;
    dist = length(diff);
    n = (dist>0.f)? diff/dist : Vec3{1,0,0};

    if(dist <= a.radius + b.radius){
        Vec3 contact = findContactPoint(a,b);
        Vec3 ra = contact - a.position;
        Vec3 rb = contact - b.position;
        Vec3 localA = rotateInv(a.orientation, ra);
        Vec3 localB = rotateInv(b.orientation, rb);
        float da = a.de ? a.de(localA / a.radius) * a.radius : 0.f;
        float db = b.de ? b.de(localB / b.radius) * b.radius : 0.f;
        const float eps = 0.001f;
        if(da < eps && db < eps){
            Vec3 va = a.velocity + cross(a.angularVelocity, ra);
            Vec3 vb = b.velocity + cross(b.angularVelocity, rb);
            float rel = dot(vb - va, n);
            if(rel < 0.0f){
                float raCn = length(cross(ra, n));
                float rbCn = length(cross(rb, n));
                float invMass = 1.0f/a.mass + 1.0f/b.mass +
                                (raCn*raCn)/a.inertia + (rbCn*rbCn)/b.inertia;
                float j = -(1.0f + 1.0f) * rel / invMass;
                Vec3 impulse = n * j;
                a.velocity = a.velocity + impulse / a.mass;
                b.velocity = b.velocity - impulse / b.mass;
                a.angularVelocity = a.angularVelocity + cross(ra, impulse)/a.inertia;
                b.angularVelocity = b.angularVelocity - cross(rb, impulse)/b.inertia;

                Vec3 rv = vb - va;
                Vec3 tangent = rv - n * dot(rv, n);
                float tlen = length(tangent);
                if(tlen > 1e-6f){
                    tangent = tangent / tlen;
                    float jt = -dot(rv, tangent) / invMass;
                    const float mu = 0.5f;
                    jt = std::clamp(jt, -j*mu, j*mu);
                    Vec3 fImpulse = tangent * jt;
                    a.velocity = a.velocity + fImpulse / a.mass;
                    b.velocity = b.velocity - fImpulse / b.mass;
                    a.angularVelocity = a.angularVelocity + cross(ra, fImpulse)/a.inertia;
                    b.angularVelocity = b.angularVelocity - cross(rb, fImpulse)/b.inertia;
                }

                // positional correction
                float pen = a.radius + b.radius - dist;
                Vec3 corr = n * (pen * 0.5f);
                a.position = a.position - corr;
                b.position = b.position + corr;
            }
        }
    }
}
