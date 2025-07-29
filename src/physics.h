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

inline bool collideSphereFractal(Vec3 C0, Vec3 V, float dt, float r,
                                 Vec3 objPos,
                                 Vec3& hitPoint, Vec3& hitNormal, float& tHit){
    float len = length(V) * dt;
    if(len <= 0.f) return false;
    Vec3 dir = normalize(V);
    float t = 0.f;
    for(int i=0;i<64 && t < len; ++i){
        Vec3 p = C0 + dir * t;
        float d = mandelbulbDE(p - objPos) - r;
        if(d < 1e-3f){
            t -= d;
            float lo = t - d, hi = t;
            for(int j=0;j<4;++j){
                float mid = 0.5f*(lo+hi);
                float dm = mandelbulbDE(C0 + dir*mid - objPos) - r;
                if(dm > 0.f) lo = mid; else hi = mid;
            }
            tHit = 0.5f*(lo+hi);
            hitPoint = C0 + dir * tHit;
            const float eps = 1e-4f;
            Vec3 ex{eps,0,0}, ey{0,eps,0}, ez{0,0,eps};
            Vec3 hp = hitPoint - objPos;
            hitNormal = normalize(Vec3{
                mandelbulbDE(hp+ex) - mandelbulbDE(hp-ex),
                mandelbulbDE(hp+ey) - mandelbulbDE(hp-ey),
                mandelbulbDE(hp+ez) - mandelbulbDE(hp-ez)
            });
            return true;
        }
        t += d;
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
        Vec3 hitP, hitN;
        float tHit;
        if(collideSphereFractal(C0, V0, dt, a.radius,
                                b.position, hitP, hitN, tHit)){
            // a) position at contact
            a.position = hitP;
            // b) impulse
            float vn = dot(V0 - b.velocity, hitN);
            if(vn < 0.0f){
                float e = 1.0f;
                float j = -(1+e)*vn / (1.0f/a.mass + 1.0f/b.mass);
                a.velocity = V0 + (j/a.mass)*hitN;
                b.velocity -=        (j/b.mass)*hitN;
            }
            // c) finish the remaining motion after collision
            float rem = dt - tHit;
            a.position += a.velocity * rem;
        } else {
            // no collision
            a.position = C0 + V0 * dt;
        }
    }

    // 3) And now B hitting A
    {
        Vec3 C0 = b.position;
        Vec3 V0 = b.velocity;
        Vec3 hitP, hitN;
        float tHit;
        if(collideSphereFractal(C0, V0, dt, b.radius,
                                a.position, hitP, hitN, tHit)){
            b.position = hitP;
            float vn = dot(V0 - a.velocity, hitN);
            if(vn < 0.0f){
                float e = 1.0f;
                float j = -(1+e)*vn / (1.0f/a.mass + 1.0f/b.mass);
                b.velocity = V0 + (j/b.mass)*hitN;
                a.velocity -=        (j/a.mass)*hitN;
            }
            float rem = dt - tHit;
            b.position += b.velocity * rem;
        } else {
            b.position = C0 + V0 * dt;
        }
    }
}
