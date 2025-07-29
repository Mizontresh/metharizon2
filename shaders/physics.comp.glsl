#version 450
#extension GL_GOOGLE_include_directive : enable
#include "fractal.glsl"
layout(local_size_x = 64) in;

struct Obj {
    vec4 posrad; // xyz position, w radius
    vec4 velmass; // xyz velocity, w mass
};

layout(std430, binding=0) buffer Spheres {
    Obj objs[];
};

layout(push_constant) uniform PC {
    float dt;
} pc;


bool collideSphereFractal(vec3 C0, vec3 delta, float r,
                          out vec3 hitP, out vec3 hitN, out float tHit){
    float len = length(delta);
    if(len < 1e-8) return false;
    vec3 dir = delta / len;
    float t = 0.0;
    for(int i=0;i<128 && t<len;i++){
        vec3 p = C0 + dir*t;
        float d = fractalDE(p) - r;
        if(d < 1e-4){
            float lo=t-d, hi=t;
            for(int j=0;j<8;j++){
                float mid=0.5*(lo+hi);
                float dm = fractalDE(C0+dir*mid) - r;
                if(dm>0) lo=mid; else hi=mid;
            }
            tHit=0.5*(lo+hi);
            hitP=C0+dir*tHit;
            float eps=1e-4;
            hitN = normalize(vec3(
                fractalDE(hitP+vec3(eps,0,0)) - fractalDE(hitP-vec3(eps,0,0)),
                fractalDE(hitP+vec3(0,eps,0)) - fractalDE(hitP-vec3(0,eps,0)),
                fractalDE(hitP+vec3(0,0,eps)) - fractalDE(hitP-vec3(0,0,eps))
            ));
            return true;
        }
        t += d;
    }
    return false;
}

void main(){
    uint i = gl_GlobalInvocationID.x;
    Obj me = objs[i];
    vec3 hitP, hitN;
    float tHit;
    vec3 pos = me.posrad.xyz;
    vec3 vel = me.velmass.xyz;
    float radius = me.posrad.w;
    float mass = me.velmass.w;
    if(collideSphereFractal(pos, vel*pc.dt, radius, hitP, hitN, tHit)){
        float travel = length(vel*pc.dt);
        float tTime = (travel>0)?(tHit/travel)*pc.dt:0.0;
        pos = hitP;
        float vn = dot(vel, hitN);
        if(vn<0.0){
            float j = -1.8*vn*mass;
            vel += (j/mass)*hitN;
        }
        pos += vel*(pc.dt - tTime);
    }else{
        pos += vel*pc.dt;
    }
    me.posrad.xyz = pos;
    me.velmass.xyz = vel;
    objs[i] = me;
}
