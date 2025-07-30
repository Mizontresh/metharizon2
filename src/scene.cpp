#include "scene.hpp"

// bring in globals from main.cpp:
extern VkDevice              device;
extern VkPhysicalDevice      physDevice;
extern VkQueue               queue;
extern uint32_t              queueFamily;
extern VkDescriptorSetLayout dsLayout;
extern VkDescriptorPool      dsPool;
extern VkExtent2D            storageExtent;

Scene::Scene(const Camera& cam)
  : _camera(cam)
{
    // two bodies, each mass=1, moving toward each other
    _bodies = {
        {
            { -1.0f, 0.0f, 3.0f, 1.0f }, // pos.xyz, mass
            {  0.5f, 0.2f, 0.0f, 0.0f }, // velocity
            {  0.0f, 0.0f, 0.0f, 0.0f }, // angVel
            {  0.0f, 0.0f, 0.0f, 1.0f }, // orient (identity)
            {  20.0f, 0.0f, 0.0f, 0.0f }  // extra.x = maxIter
        },
        {
            {  1.0f, 0.0f, 3.0f, 1.0f },
            { -0.5f,-0.2f, 0.0f, 0.0f },
            {  0.0f, 0.0f, 0.0f, 0.0f },
            {  0.0f, 0.0f, 0.0f, 1.0f },
            {  20.0f, 0.0f, 0.0f, 0.0f }
        }
    };

    // initialize the physics module once:
    PhysicsModule::init(
        device, physDevice, queue,
        queueFamily,
        storageExtent
    );

    PhysicsModule::uploadBodies(_bodies);
}

void Scene::updateAndDispatch(VkCommandBuffer cmd)
{
    // update camera and record compute passes
    PhysicsModule::step(_camera);
    PhysicsModule::recordDispatch(cmd);
}
