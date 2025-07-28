// main.cpp
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <array>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <set>
#include <algorithm>
#include <cmath>

const uint32_t WIDTH  = 800;
const uint32_t HEIGHT = 600;

#define VK_CHECK(fn)                                                            \
    do {                                                                        \
        VkResult res = (fn);                                                    \
        if (res != VK_SUCCESS)                                                  \
            throw std::runtime_error(std::string("Vulkan error at ") + #fn);    \
    } while (0)

struct Camera {
    alignas(16) float pos[3];
    alignas(16) float forward[3];
    alignas(16) float up[3];
    alignas(16) float right[3];
};

VkInstance            instance;
GLFWwindow*           window;
VkSurfaceKHR          surface;
VkPhysicalDevice      physDevice;
VkDevice              device;
VkQueue               compQueue;
uint32_t              compQueueFamily;

VkSwapchainKHR        swapchain;
VkFormat              swapchainFormat;
std::vector<VkImage>        swapchainImages;
std::vector<VkImageView>    swapchainViews;

VkCommandPool         cmdPool;
VkCommandBuffer       cmdBuf;

VkDescriptorSetLayout dsLayout;
VkDescriptorPool      dsPool;
VkDescriptorSet       ds;

VkPipelineLayout      pipelineLayout;
VkPipeline            computePipeline;
VkShaderModule        compShader;

VkExtent2D            extent;
VkImage               storageImage;
VkDeviceMemory        storageMem;
VkImageView           storageView;

VkBuffer              cameraBuffer;
VkDeviceMemory        cameraMem;
VkDescriptorBufferInfo cameraBufferInfo;
VkDescriptorImageInfo  storageImageInfo;

auto now = [](){ return std::chrono::high_resolution_clock::now(); };

// find first compute queue
uint32_t findComputeQueueFamily(VkPhysicalDevice pd) {
    uint32_t cnt = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &cnt, nullptr);
    std::vector<VkQueueFamilyProperties> qf(cnt);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &cnt, qf.data());
    for (uint32_t i = 0; i < cnt; i++)
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
            return i;
    throw std::runtime_error("No compute queue");
}

void pickPhysicalDevice() {
    uint32_t cnt = 0;
    vkEnumeratePhysicalDevices(instance, &cnt, nullptr);
    if (!cnt) throw std::runtime_error("No GPU");
    std::vector<VkPhysicalDevice> devs(cnt);
    vkEnumeratePhysicalDevices(instance, &cnt, devs.data());
    for (auto &d: devs) {
        try {
            compQueueFamily = findComputeQueueFamily(d);
            physDevice = d;
            return;
        } catch(...) {}
    }
    throw std::runtime_error("No suitable GPU");
}

void createInstance() {
    if (!glfwInit())
        throw std::runtime_error("GLFW init failed");
    if (!glfwVulkanSupported())
        throw std::runtime_error("GLFW Vulkan unsupported");
    // app info
    VkApplicationInfo app{};
    app.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName   = "Compute Raymarch";
    app.apiVersion         = VK_API_VERSION_1_1;
    // required extensions
    uint32_t extCount = 0;
    const char** exts = glfwGetRequiredInstanceExtensions(&extCount);
    VkInstanceCreateInfo ci{};
    ci.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo        = &app;
    ci.enabledExtensionCount   = extCount;
    ci.ppEnabledExtensionNames = exts;
    VK_CHECK(vkCreateInstance(&ci, nullptr, &instance));
}

void createWindowAndSurface() {
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(WIDTH, HEIGHT, "Raymarch", nullptr, nullptr);
    if (!window) throw std::runtime_error("GLFW window failed");
    // mouse
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    if (glfwRawMouseMotionSupported())
        glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
    VK_CHECK(glfwCreateWindowSurface(instance, window, nullptr, &surface));
}

void createDeviceAndQueue() {
    // priority for our single queue
    float queuePriority = 1.0f;

    // describe the single compute+present queue we want
    VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfo.queueFamilyIndex = compQueueFamily;
    queueInfo.queueCount       = 1;
    queueInfo.pQueuePriorities = &queuePriority;

    // request the VK_KHR_swapchain extension so vkCreateSwapchainKHR will be available
    const char* deviceExtensions[] = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    VkDeviceCreateInfo createInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    createInfo.queueCreateInfoCount    = 1;
    createInfo.pQueueCreateInfos       = &queueInfo;
    createInfo.enabledExtensionCount   = 1;
    createInfo.ppEnabledExtensionNames = deviceExtensions;

    // create the logical device
    VK_CHECK(vkCreateDevice(physDevice, &createInfo, nullptr, &device));

    // grab the handle to our compute/present queue
    vkGetDeviceQueue(device, compQueueFamily, 0, &compQueue);
}


void createSwapchain() {
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physDevice, surface, &caps);
    extent = caps.currentExtent;
    uint32_t fc=0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physDevice, surface, &fc, nullptr);
    std::vector<VkSurfaceFormatKHR> fmts(fc);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physDevice, surface, &fc, fmts.data());
    swapchainFormat = fmts[0].format;
    VkSwapchainCreateInfoKHR sci{};
    sci.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    sci.surface          = surface;
    sci.minImageCount    = caps.minImageCount+1;
    sci.imageFormat      = swapchainFormat;
    sci.imageColorSpace  = fmts[0].colorSpace;
    sci.imageExtent      = extent;
    sci.imageArrayLayers = 1;
    sci.imageUsage       = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sci.preTransform     = caps.currentTransform;
    sci.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sci.presentMode      = VK_PRESENT_MODE_FIFO_KHR;
    sci.clipped          = VK_TRUE;
    VK_CHECK(vkCreateSwapchainKHR(device, &sci, nullptr, &swapchain));
    uint32_t ic=0;
    vkGetSwapchainImagesKHR(device, swapchain, &ic, nullptr);
    swapchainImages.resize(ic);
    vkGetSwapchainImagesKHR(device, swapchain, &ic, swapchainImages.data());
    swapchainViews.resize(ic);
    for (uint32_t i = 0; i < ic; i++) {
        VkImageViewCreateInfo ivci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        ivci.image            = swapchainImages[i];
        ivci.viewType         = VK_IMAGE_VIEW_TYPE_2D;
        ivci.format           = swapchainFormat;
        ivci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
        VK_CHECK(vkCreateImageView(device, &ivci, nullptr, &swapchainViews[i]));
    }
}

void createCommandPoolAndBuffer() {
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = compQueueFamily;
    VK_CHECK(vkCreateCommandPool(device,&cpi,nullptr,&cmdPool));
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool        = cmdPool;
    cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(device,&cbai,&cmdBuf));
}

void createStorageImage() {
    extent = { WIDTH, HEIGHT };
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.format        = VK_FORMAT_R8G8B8A8_UNORM;
    ici.extent        = {extent.width,extent.height,1};
    ici.mipLevels     = 1;
    ici.arrayLayers   = 1;
    ici.usage         = VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    VK_CHECK(vkCreateImage(device,&ici,nullptr,&storageImage));
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(device,storageImage,&mr);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(physDevice,&mp);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = mr.size;
    for (uint32_t i=0;i<mp.memoryTypeCount;i++){
        if ((mr.memoryTypeBits&(1<<i)) &&
            (mp.memoryTypes[i].propertyFlags&VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
             ==VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
            mai.memoryTypeIndex = i;
            break;
        }
    }
    VK_CHECK(vkAllocateMemory(device,&mai,nullptr,&storageMem));
    VK_CHECK(vkBindImageMemory(device,storageImage,storageMem,0));
    VkImageViewCreateInfo ivci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    ivci.image            = storageImage;
    ivci.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    ivci.format           = ici.format;
    ivci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
    VK_CHECK(vkCreateImageView(device,&ivci,nullptr,&storageView));
}

void createCameraBuffer() {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size  = sizeof(Camera);
    bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    VK_CHECK(vkCreateBuffer(device,&bci,nullptr,&cameraBuffer));
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(device,cameraBuffer,&mr);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(physDevice,&mp);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = mr.size;
    for (uint32_t i=0;i<mp.memoryTypeCount;i++){
        if ((mr.memoryTypeBits&(1<<i)) &&
            (mp.memoryTypes[i].propertyFlags&VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
             ==VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
            mai.memoryTypeIndex = i;
            break;
        }
    }
    VK_CHECK(vkAllocateMemory(device,&mai,nullptr,&cameraMem));
    VK_CHECK(vkBindBufferMemory(device,cameraBuffer,cameraMem,0));
    // <— assign fields individually, not with braces
    cameraBufferInfo.buffer = cameraBuffer;
    cameraBufferInfo.offset = 0;
    cameraBufferInfo.range  = sizeof(Camera);
}

void createDescriptorSet() {
    VkDescriptorSetLayoutBinding b0{0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutBinding b1{1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT};
    std::array<VkDescriptorSetLayoutBinding,2> binds = {b0,b1};
    VkDescriptorSetLayoutCreateInfo dsli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dsli.bindingCount = 2;
    dsli.pBindings    = binds.data();
    VK_CHECK(vkCreateDescriptorSetLayout(device,&dsli,nullptr,&dsLayout));
    VkDescriptorPoolSize ps0{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1};
    VkDescriptorPoolSize ps1{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1};
    std::array<VkDescriptorPoolSize,2> pss = {ps0,ps1};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 2;
    dpci.pPoolSizes    = pss.data();
    VK_CHECK(vkCreateDescriptorPool(device,&dpci,nullptr,&dsPool));
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool     = dsPool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &dsLayout;
    VK_CHECK(vkAllocateDescriptorSets(device,&dsai,&ds));
    // <— assign fields individually, not with braces
    storageImageInfo.sampler     = VK_NULL_HANDLE;
    storageImageInfo.imageView   = storageView;
    storageImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet w0{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,nullptr,ds,0,0,1,
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,&storageImageInfo,nullptr};
    VkWriteDescriptorSet w1{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,nullptr,ds,1,0,1,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,nullptr,&cameraBufferInfo,nullptr};
    std::array<VkWriteDescriptorSet,2> writes = {w0,w1};
    vkUpdateDescriptorSets(device,2,writes.data(),0,nullptr);
}

void createComputePipeline() {
    std::vector<char> spv;
    FILE* f = fopen("../shaders/comp.spv","rb");
    fseek(f,0,SEEK_END); size_t n=ftell(f); rewind(f);
    spv.resize(n); fread(spv.data(),1,n,f); fclose(f);
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = spv.size();
    smci.pCode    = reinterpret_cast<const uint32_t*>(spv.data());
    VK_CHECK(vkCreateShaderModule(device,&smci,nullptr,&compShader));

    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts    = &dsLayout;
    VK_CHECK(vkCreatePipelineLayout(device,&plci,nullptr,&pipelineLayout));

    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = compShader;
    cpci.stage.pName  = "main";
    cpci.layout       = pipelineLayout;
    VK_CHECK(vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&cpci,nullptr,&computePipeline));
}

void recordCompute() {
    vkResetCommandBuffer(cmdBuf,0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VK_CHECK(vkBeginCommandBuffer(cmdBuf,&bi));

    VkImageMemoryBarrier bar0{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    bar0.oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED;
    bar0.newLayout        = VK_IMAGE_LAYOUT_GENERAL;
    bar0.image            = storageImage;
    bar0.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
    bar0.dstAccessMask    = VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmdBuf,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,0,nullptr,0,nullptr,1,&bar0);

    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipelineLayout,0,1,&ds,0,nullptr);
    vkCmdDispatch(cmdBuf,
        (extent.width+15)/16, (extent.height+15)/16, 1);

    VkImageMemoryBarrier bar1 = bar0;
    bar1.oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
    bar1.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    bar1.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bar1.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmdBuf,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,0,nullptr,0,nullptr,1,&bar1);

    VK_CHECK(vkEndCommandBuffer(cmdBuf));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cmdBuf;
    VK_CHECK(vkQueueSubmit(compQueue,1,&si,VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(compQueue));
}

void drawFrame() {
    recordCompute();

    uint32_t idx;
    VK_CHECK(vkAcquireNextImageKHR(device,swapchain,UINT64_MAX,VK_NULL_HANDLE,VK_NULL_HANDLE,&idx));

    vkResetCommandBuffer(cmdBuf,0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VK_CHECK(vkBeginCommandBuffer(cmdBuf,&bi));

    VkImageMemoryBarrier dstBar{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    dstBar.oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED;
    dstBar.newLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    dstBar.image            = swapchainImages[idx];
    dstBar.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
    dstBar.dstAccessMask    = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmdBuf,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,0,nullptr,0,nullptr,1,&dstBar);

    VkImageBlit blit{};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT,0,0,1};
    blit.srcOffsets[1]  = {int32_t(extent.width),int32_t(extent.height),1};
    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT,0,0,1};
    blit.dstOffsets[1]  = blit.srcOffsets[1];
    vkCmdBlitImage(cmdBuf,
        storageImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        swapchainImages[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,&blit,VK_FILTER_NEAREST);

    VkImageMemoryBarrier presBar = dstBar;
    presBar.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    presBar.newLayout     = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    presBar.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    presBar.dstAccessMask = 0;
    vkCmdPipelineBarrier(cmdBuf,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0,0,nullptr,0,nullptr,1,&presBar);

    VK_CHECK(vkEndCommandBuffer(cmdBuf));

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cmdBuf;
    VK_CHECK(vkQueueSubmit(compQueue,1,&si,VK_NULL_HANDLE));

    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.swapchainCount = 1;
    pi.pSwapchains    = &swapchain;
    pi.pImageIndices  = &idx;
    VK_CHECK(vkQueuePresentKHR(compQueue,&pi));
    VK_CHECK(vkQueueWaitIdle(compQueue));
}

void mainLoop() {
    Camera cam{};
    cam.pos[0]=0; cam.pos[1]=0; cam.pos[2]=3;
    cam.forward[0]=0; cam.forward[1]=0; cam.forward[2]=-1;
    cam.up[0]=0;    cam.up[1]=1;    cam.up[2]=0;
    cam.right[0]=1; cam.right[1]=0; cam.right[2]=0;
    double lastX=WIDTH/2.0, lastY=HEIGHT/2.0;
    glfwSetCursorPos(window,lastX,lastY);

    while(!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        double mx,my;
        glfwGetCursorPos(window,&mx,&my);
        float dx=float(mx-lastX), dy=float(my-lastY);
        lastX=mx; lastY=my;

        const float sens=0.0025f;
        float yaw   = -dx*sens;
        float pitch = -dy*sens;

        auto rotate = [&](float a, float axis[3], float v[3]){
            float c=std::cos(a), s=std::sin(a);
            float x=axis[0],y=axis[1],z=axis[2];
            float M[3][3] = {
                {c+(1-c)*x*x,    (1-c)*x*y - s*z, (1-c)*x*z + s*y},
                {(1-c)*y*x + s*z, c+(1-c)*y*y,    (1-c)*y*z - s*x},
                {(1-c)*z*x - s*y, (1-c)*z*y + s*x, c+(1-c)*z*z}
            };
            float r[3];
            for(int i=0;i<3;i++) r[i]=M[i][0]*v[0]+M[i][1]*v[1]+M[i][2]*v[2];
            std::memcpy(v,r,sizeof(r));
        };
        rotate(yaw, cam.up,    cam.forward);
        rotate(yaw, cam.up,    cam.right);
        rotate(pitch,cam.right,cam.forward);

        auto cross = [](float a[3], float b[3], float out[3]){
            out[0]=a[1]*b[2]-a[2]*b[1];
            out[1]=a[2]*b[0]-a[0]*b[2];
            out[2]=a[0]*b[1]-a[1]*b[0];
        };
        cross(cam.forward, cam.up, cam.right);

        void* ptr;
        vkMapMemory(device,cameraMem,0,sizeof(cam),0,&ptr);
        std::memcpy(ptr,&cam,sizeof(cam));
        vkUnmapMemory(device,cameraMem);

        drawFrame();
    }
}

void cleanup() {
    vkDeviceWaitIdle(device);
    vkDestroyPipeline(device,computePipeline,nullptr);
    vkDestroyPipelineLayout(device,pipelineLayout,nullptr);
    vkDestroyShaderModule(device,compShader,nullptr);
    vkDestroyDescriptorPool(device,dsPool,nullptr);
    vkDestroyDescriptorSetLayout(device,dsLayout,nullptr);
    vkDestroyBuffer(device,cameraBuffer,nullptr);
    vkFreeMemory(device,cameraMem,nullptr);
    vkDestroyImageView(device,storageView,nullptr);
    vkDestroyImage(device,storageImage,nullptr);
    vkFreeMemory(device,storageMem,nullptr);
    for (auto v: swapchainViews) vkDestroyImageView(device,v,nullptr);
    vkDestroySwapchainKHR(device,swapchain,nullptr);
    vkDestroyCommandPool(device,cmdPool,nullptr);
    vkDestroyDevice(device,nullptr);
    vkDestroySurfaceKHR(instance,surface,nullptr);
    vkDestroyInstance(instance,nullptr);
    glfwDestroyWindow(window);
    glfwTerminate();
}

int main(){
    try {
        createInstance();
        createWindowAndSurface();
        pickPhysicalDevice();
        createDeviceAndQueue();
        createSwapchain();
        createCommandPoolAndBuffer();
        createStorageImage();
        createCameraBuffer();
        createDescriptorSet();
        createComputePipeline();
        mainLoop();
        cleanup();
    } catch(const std::exception &e){
        std::cerr<<"Fatal: "<<e.what()<<"\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
