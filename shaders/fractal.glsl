#ifndef FRACTAL_GLSL
#define FRACTAL_GLSL

const int   MAX_ITER   = 12;
const float BAILOUT    = 4.0;
const float POWER      = 8.0;
const float STEP_EPS   = 1e-4;

float fractalDE(in vec3 p) {
    vec4 z = vec4(0.0, p);
    const vec4 C = vec4(-0.2, 0.7, 0.0, 0.0);
    float dr = 1.0;
    float r  = 0.0;
    for(int i=0;i<MAX_ITER;i++){
        r = length(z);
        if(r>BAILOUT) break;
        float theta = acos(z.w/r);
        float phi   = atan(length(z.yzw), z.x);
        float psi   = atan(z.y, z.z);
        dr = POWER * pow(r, POWER-1.0) * dr + 1.0;
        float rp = pow(r, POWER);
        theta *= POWER; phi *= POWER; psi *= POWER;
        z = rp * vec4(cos(theta),
                     sin(theta)*sin(phi)*cos(psi),
                     sin(theta)*sin(phi)*sin(psi),
                     sin(theta)*cos(phi)) + C;
    }
    return length(z)/dr;
}

float sphereMarch(in vec3 ro, in vec3 rd) {
    float t = 0.0;
    for(int i=0;i<128;i++){
        vec3 p = ro + rd*t;
        float d = fractalDE(p);
        if(d < STEP_EPS) break;
        t += d;
        if(t>50.0) break;
    }
    return t;
}

#endif
