#pragma once
#include <cmath>

struct Vec3 {
    float x, y, z;
};

inline Vec3 operator+(Vec3 a, Vec3 b){ return {a.x+b.x, a.y+b.y, a.z+b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b){ return {a.x-b.x, a.y-b.y, a.z-b.z}; }
inline Vec3 operator*(Vec3 a, float s){ return {a.x*s, a.y*s, a.z*s}; }
inline Vec3 operator/(Vec3 a, float s){ return {a.x/s, a.y/s, a.z/s}; }
inline Vec3 operator*(float s, Vec3 a){ return {a.x*s, a.y*s, a.z*s}; }
inline Vec3& operator+=(Vec3& a, Vec3 b){ a.x+=b.x; a.y+=b.y; a.z+=b.z; return a; }
inline Vec3& operator-=(Vec3& a, Vec3 b){ a.x-=b.x; a.y-=b.y; a.z-=b.z; return a; }
inline float dot(Vec3 a, Vec3 b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
inline float length(Vec3 a){ return std::sqrt(dot(a,a)); }
inline Vec3 normalize(Vec3 a){ float l = length(a); return (l>0)? a/l : Vec3{0,0,0}; }
inline Vec3 cross(Vec3 a, Vec3 b){
    return { a.y*b.z - a.z*b.y,
             a.z*b.x - a.x*b.z,
             a.x*b.y - a.y*b.x };
}

struct FractalObject {
    Vec3 position;
    Vec3 velocity;
    float radius;
    float mass;
};

inline float mandelbulbDE(Vec3 p){
    Vec3 z = p;
    float dr = 1.0f;
    float r  = 0.0f;
    const int ITER = 12;
    for(int i=0;i<ITER;i++){
        r = length(z);
        if(r>2.0f) break;
        float theta = std::acos(z.z/r);
        float phi   = std::atan2(z.y, z.x);
        dr = std::pow(r,7.0f)*8.0f*dr + 1.0f;
        float zr = std::pow(r,8.0f);
        theta *= 8.0f;
        phi   *= 8.0f;
        Vec3 nz{ std::sin(theta)*std::cos(phi),
                 std::sin(theta)*std::sin(phi),
                 std::cos(theta) };
        z = zr*nz + p;
    }
    return 0.5f*std::log(r)*r/dr;
}

// pass the *movement delta* directly instead of V+dt separately
inline bool collideSphereFractal(
    const Vec3 &C0,         // start position of the sphere
    const Vec3 &delta,      // full displacement = velocity * dt
    float       r,          // sphere radius
    const Vec3 &objPos,     // fractal center
    Vec3       &hitPoint,
    Vec3       &hitNormal,
    float      &tHit
){
    float len = length(delta);
    if(len < 1e-8f) return false;
    Vec3 dir = delta / len;
    float t = 0.f;
    for(int i = 0; i < 128 && t < len; ++i) {
        Vec3 p = C0 + dir * t;
        // distance from your *moving* sphere surface to the fractal:
        float d = mandelbulbDE(p - objPos) - r;
        if(d < 1e-4f) {
            // back-off and binary-search on [t-d, t]
            float lo = t - d, hi = t;
            for(int j = 0; j < 8; ++j) {
                float mid = 0.5f * (lo + hi);
                float dm  = mandelbulbDE(C0 + dir*mid - objPos) - r;
                if(dm > 0) lo = mid; else hi = mid;
            }
            tHit    = 0.5f*(lo + hi);
            hitPoint = C0 + dir * tHit;
            // compute normal from fine gradient
            const float eps = 1e-5f;
            Vec3 ex{eps,0,0}, ey{0,eps,0}, ez{0,0,eps};
            Vec3 rel = hitPoint - objPos;
            hitNormal = normalize(Vec3{
              mandelbulbDE(rel+ex) - mandelbulbDE(rel-ex),
              mandelbulbDE(rel+ey) - mandelbulbDE(rel-ey),
              mandelbulbDE(rel+ez) - mandelbulbDE(rel-ez)
            });
            return true;
        }
        t += d;  // sphere-march step
    }
    return false;
}
// March from O along unit dir until fractal surface reached
inline float sphereMarchSurface(Vec3 O, Vec3 dir){
    float t=0.f;
    const float MAX_EXTENT=5.0f;
    const float STEP_EPS=1e-4f;
    for(int i=0;i<128;i++){
        Vec3 p = O + dir*t;
        float dist = mandelbulbDE(p);
        if(dist < STEP_EPS) return t;
        t += dist;
        if(t>MAX_EXTENT) break;
    }
    return 1e6f;
}

// narrow phase check between two fractal centers
inline bool collideFractalFractal(const FractalObject &A, const FractalObject &B,
                                  Vec3 &contact, Vec3 &normal, float &penetration){
    Vec3 diff = {B.position.x - A.position.x,
                 B.position.y - A.position.y,
                 B.position.z - A.position.z};
    float dist = length(diff);
    if(dist < 1e-6f) return false;
    Vec3 n = diff / dist;
    float tA = sphereMarchSurface(A.position, n);
    float tB = sphereMarchSurface(B.position, n*-1.0f);
    if(tA + tB >= dist){
        Vec3 contactA = A.position + n * tA;
        Vec3 contactB = B.position - n * tB;
        contact = (contactA + contactB) / 2.f;
        normal = n;
        penetration = tA + tB - dist;
        return true;
    }
    return false;
}

inline void stepPhysics(FractalObject &a, FractalObject &b, float dt, float G=1.0f){
    // gravity
    Vec3 diff = {b.position.x - a.position.x,
                 b.position.y - a.position.y,
                 b.position.z - a.position.z};
    float d2 = dot(diff, diff) + 1e-6f;
    float force = G * a.mass * b.mass / d2;
    Vec3 n = normalize(diff);
    Vec3 F = n * force;
    a.velocity += (F / a.mass) * dt;
    b.velocity -= (F / b.mass) * dt;

    // integrate
    a.position += a.velocity * dt;
    b.position += b.velocity * dt;

    // narrow-phase collision between fractal surfaces
    Vec3 contact, normal;
    float penetration;
    if(collideFractalFractal(a, b, contact, normal, penetration)){
        Vec3 vRel = {a.velocity.x - b.velocity.x,
                     a.velocity.y - b.velocity.y,
                     a.velocity.z - b.velocity.z};
        float vN = dot(vRel, normal);
        if(vN < 0.0f){
            float e = 1.0f;
            float j = -(1.0f+e)*vN / (1.0f/a.mass + 1.0f/b.mass);
            a.velocity += (j/a.mass) * normal;
            b.velocity -= (j/b.mass) * normal;
        }
        a.position -= normal * (penetration*0.5f);
        b.position += normal * (penetration*0.5f);
    }
}

