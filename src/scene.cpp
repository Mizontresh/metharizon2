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
        { { -1.0f, 0.0f, 3.0f, 0.0f }, {  0.5f,  0.2f, 0.0f, 1.0f } },
        { {  1.0f, 0.0f, 3.0f, 0.0f }, { -0.5f, -0.2f, 0.0f, 1.0f } }
    };

    // initialize the physics module once:
    PhysicsModule::init(
        device, physDevice, queue,
        queueFamily, dsLayout, dsPool,
        storageExtent
    );
}

void Scene::updateAndDispatch(VkCommandBuffer cmd)
{
    // upload bodies & camera, run physics, then record into cmd for ray‑march
    PhysicsModule::step(_bodies, _camera);
    PhysicsModule::recordDispatch(cmd);
}
