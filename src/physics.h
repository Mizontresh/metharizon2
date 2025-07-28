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
inline float dot(Vec3 a, Vec3 b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
inline float length(Vec3 a){ return std::sqrt(dot(a,a)); }
inline Vec3 normalize(Vec3 a){ float l = length(a); return (l>0)? a/l : Vec3{0,0,0}; }

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
    const int ITER = 8;
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

inline void stepPhysics(FractalObject& a, FractalObject& b, float dt, float G=1.0f){
    Vec3 diff = b.position - a.position;
    float distSq = dot(diff,diff) + 1e-6f;
    float dist = std::sqrt(distSq);
    Vec3 n = diff / dist;
    float forceMag = G * a.mass * b.mass / distSq;
    Vec3 force = n * forceMag;
    a.velocity = a.velocity + force / a.mass * dt;
    b.velocity = b.velocity - force / b.mass * dt;
    a.position = a.position + a.velocity * dt;
    b.position = b.position + b.velocity * dt;
    if(dist <= a.radius + b.radius){
        Vec3 mid = (a.position + b.position) * 0.5f;
        float da = mandelbulbDE(mid - a.position);
        float db = mandelbulbDE(mid - b.position);
        if(da < 0.001f && db < 0.001f){
            float rel = dot(b.velocity - a.velocity, n);
            if(rel < 0.0f){
                float j = -(1.0f + 1.0f) * rel / (1.0f/a.mass + 1.0f/b.mass);
                Vec3 impulse = n * j;
                a.velocity = a.velocity + impulse / a.mass;
                b.velocity = b.velocity - impulse / b.mass;
            }
        }
    }
}
