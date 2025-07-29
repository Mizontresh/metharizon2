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

inline void stepPhysics(FractalObject &a, FractalObject &b, float dt, float G=1.0f){
    // 1) Gravity impulse
    Vec3 diff   = b.position - a.position;
    float d2    = dot(diff, diff) + 1e-6f;
    float force = G * a.mass * b.mass / d2;
    Vec3 n      = normalize(diff);
    Vec3 F      = n * force;

    a.velocity += ( F / a.mass) * dt;
    b.velocity -= ( F / b.mass) * dt;

    // 2) Continuous collision+integration for A hitting B
    {
        Vec3 C0 = a.position;
        Vec3 V0 = a.velocity;
        Vec3 deltaA = a.velocity * dt;
        Vec3 hitP, hitN;
        float tHit;
        if(collideSphereFractal(C0, deltaA, a.radius,
                                b.position, hitP, hitN, tHit)){
            // tHit is in [0,len], so convert back to time
            float lenA = length(deltaA);
            float tTime = (lenA > 0)
                          ? (tHit / lenA) * dt
                          : 0.0f;
            // snap to contact
            a.position = C0 + normalize(deltaA) * tHit;
            // b) impulse
            float vn = dot(V0 - b.velocity, hitN);
            if(vn < 0.0f){
                float e = 1.0f;
                float j = -(1+e)*vn / (1.0f/a.mass + 1.0f/b.mass);
                a.velocity = V0 + (j/a.mass)*hitN;
                b.velocity -=        (j/b.mass)*hitN;
            }
            // c) finish the remaining motion after collision
            a.position += a.velocity * (dt - tTime);
        } else {
            // no collision
            a.position = C0 + deltaA;
        }
    }

    // 3) And now B hitting A
    {
        Vec3 C0 = b.position;
        Vec3 V0 = b.velocity;
        Vec3 deltaB = b.velocity * dt;
        Vec3 hitP, hitN;
        float tHit;
        if(collideSphereFractal(C0, deltaB, b.radius,
                                a.position, hitP, hitN, tHit)){
            float lenB = length(deltaB);
            float tTime = (lenB > 0)
                          ? (tHit / lenB) * dt
                          : 0.0f;
            b.position = C0 + normalize(deltaB) * tHit;
            float vn = dot(V0 - a.velocity, hitN);
            if(vn < 0.0f){
                float e = 1.0f;
                float j = -(1+e)*vn / (1.0f/a.mass + 1.0f/b.mass);
                b.velocity = V0 + (j/b.mass)*hitN;
                a.velocity -=        (j/a.mass)*hitN;
            }
            b.position += b.velocity * (dt - tTime);
        } else {
            b.position = C0 + deltaB;
        }
    }
}
