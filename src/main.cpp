// src/main.cpp
#define GLFW_INCLUDE_VULKAN
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#if defined(_WIN32)
#  include <windows.h>
#elif defined(__linux__)
#  include <unistd.h>
#  include <limits.h>
#endif
#include <optional>
#include <stdexcept>
#include <vector>
#include <cmath>
#include "physics.h"

const uint32_t WIDTH  = 800;
const uint32_t HEIGHT = 600;

// Simple error check macro
#define VK_CHECK(fn)                                                           \
    do {                                                                        \
        VkResult _res = (fn);                                                   \
        if (_res != VK_SUCCESS)                                                 \
            throw std::runtime_error(std::string("Vulkan error at ") + #fn);    \
    } while (0)

struct Camera {
    alignas(16) float pos[3];
    alignas(16) float forward[3];
    alignas(16) float up[3];
    alignas(16) float right[3];
};


static const float BASE_FORWARD[3] = {0.f, 0.f, -1.f};
static const float BASE_UP[3]      = {0.f, 1.f, 0.f};
static const float BASE_RIGHT[3]   = {1.f, 0.f, 0.f};

static auto now = [](){
    return std::chrono::high_resolution_clock::now();
};

bool fullscreen = false;
int windowX, windowY;
int windowW = WIDTH, windowH = HEIGHT;
bool headless = false;
VkBuffer headlessBuffer = VK_NULL_HANDLE;
VkDeviceMemory headlessMemory = VK_NULL_HANDLE;
bool screenshotPending = false;
uint32_t screenshotIndex = 0;

void keyCallback(GLFWwindow* win, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(win, GLFW_TRUE);
    if (key == GLFW_KEY_F11 && action == GLFW_PRESS) {
        fullscreen = !fullscreen;
        if (fullscreen) {
            glfwGetWindowPos(win, &windowX, &windowY);
            glfwGetWindowSize(win, &windowW, &windowH);
            GLFWmonitor* mon = glfwGetPrimaryMonitor();
            const GLFWvidmode* mode = glfwGetVideoMode(mon);
            glfwSetWindowMonitor(win, mon, 0, 0, mode->width, mode->height, mode->refreshRate);
        } else {
            glfwSetWindowMonitor(win, nullptr, windowX, windowY, windowW, windowH, 0);
        }
    } else if(key == GLFW_KEY_P && action == GLFW_PRESS){
        screenshotPending = true;
    }
}

GLFWwindow*           window;
VkInstance            instance;
VkSurfaceKHR          surface;
VkPhysicalDevice      physDevice = VK_NULL_HANDLE;
VkDevice              device;
VkQueue               queue;
uint32_t              queueFamily;

void createHeadlessBuffer();
void destroyHeadlessBuffer();
void drawFrameHeadless(uint32_t frame, Camera& cam, const FractalObject& a, const FractalObject& b);

VkSwapchainKHR        swapchain;
VkFormat              swapchainFormat;
VkExtent2D            swapchainExtent;
std::vector<VkImage>         swapImages;
std::vector<VkImageView>     swapImageViews;

VkImage               storageImage;
VkDeviceMemory        storageMemory;
VkImageView           storageView;
VkExtent2D            storageExtent;

VkBuffer              cameraBuffer;
VkDeviceMemory        cameraMemory;
VkDescriptorBufferInfo cameraBufferInfo;

VkBuffer              objectBuffer;
VkDeviceMemory        objectMemory;
VkDescriptorBufferInfo objectBufferInfo;

VkDescriptorSetLayout dsLayout;
VkDescriptorPool      dsPool;
VkDescriptorSet       ds;
VkDescriptorImageInfo storageImageInfo;

VkPipelineLayout      pipelineLayout;
VkPipeline            pipeline;
VkShaderModule        compShader;

VkCommandPool         cmdPool;
std::vector<VkCommandBuffer> cmdBuffers;

VkSemaphore           semImageAvailable;
VkSemaphore           semRenderFinished;

//
// Helpers
//

static std::vector<char> readFile(const std::string &path) {
    std::ifstream f(path, std::ios::ate | std::ios::binary);
    if (!f) throw std::runtime_error("Failed to open " + path);
    size_t sz = (size_t)f.tellg();
    std::vector<char> buf(sz);
    f.seekg(0);
    f.read(buf.data(), sz);
    return buf;
}

static std::string getExecutableDir() {
#if defined(_WIN32)
    char path[MAX_PATH];
    DWORD len = GetModuleFileNameA(nullptr, path, MAX_PATH);
    if (len == 0 || len == MAX_PATH)
        return ".";
    std::string exePath(path, len);
    size_t pos = exePath.find_last_of("/\\");
    return exePath.substr(0, pos);
#elif defined(__linux__)
    char path[1024];
    ssize_t len = readlink("/proc/self/exe", path, sizeof(path)-1);
    if (len == -1)
        return ".";
    path[len] = '\0';
    std::string exePath(path);
    size_t pos = exePath.find_last_of('/');
    return exePath.substr(0, pos);
#else
    return ".";
#endif
}

// Find a queue family that supports both compute & present
void pickPhysicalDevice() {
    uint32_t devCount = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &devCount, nullptr));
    if (!devCount) throw std::runtime_error("No GPU found");
    std::vector<VkPhysicalDevice> devs(devCount);
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &devCount, devs.data()));

    for (auto &dev : devs) {
        uint32_t qCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qCount, nullptr);
        std::vector<VkQueueFamilyProperties> qProps(qCount);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qCount, qProps.data());

        for (uint32_t i = 0; i < qCount; i++) {
            bool hasCompute = qProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT;
            VkBool32 presentCap = VK_FALSE;
            if(!headless)
                vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, surface, &presentCap);
            if (hasCompute && (headless || presentCap)) {
                physDevice    = dev;
                queueFamily   = i;
                return;
            }
        }
    }
    throw std::runtime_error("No GPU queue supports both compute & present");
}

void createInstance() {
    const char* displayEnv = getenv("DISPLAY");
    const char* wlDisplayEnv = getenv("WAYLAND_DISPLAY");
#if defined(__linux__) && defined(GLFW_PLATFORM_NULL)
    if ((!displayEnv || !*displayEnv) && (!wlDisplayEnv || !*wlDisplayEnv)) {
        glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_NULL);
        headless = true;
    }
#endif
    if (!glfwInit())
        throw std::runtime_error("GLFW init failed");
    if (!glfwVulkanSupported())
        throw std::runtime_error("Vulkan not supported by GLFW");

    VkApplicationInfo appInfo{};
    appInfo.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName   = "ComputeRaymarch";
    appInfo.apiVersion         = VK_API_VERSION_1_1;

    uint32_t extCount = 0;
    const char** glfwExts = glfwGetRequiredInstanceExtensions(&extCount);

    VkInstanceCreateInfo ci{};
    ci.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo        = &appInfo;
    ci.enabledExtensionCount   = extCount;
    ci.ppEnabledExtensionNames = glfwExts;

    VK_CHECK(vkCreateInstance(&ci, nullptr, &instance));
}

void createWindowAndSurface() {
    if(headless){
        window = nullptr;
        surface = VK_NULL_HANDLE;
        return;
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(WIDTH, HEIGHT, "Raymarch", nullptr, nullptr);
    if (!window) throw std::runtime_error("Failed to create GLFW window");
    glfwSetKeyCallback(window, keyCallback);
    VK_CHECK(glfwCreateWindowSurface(instance, window, nullptr, &surface));
}

void createLogicalDeviceAndQueue() {
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = queueFamily;
    qci.queueCount       = 1;
    qci.pQueuePriorities = &prio;

    // Enable swapchain extension so we can present
    const char* devExts[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

    VkDeviceCreateInfo di{};
    di.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    di.queueCreateInfoCount    = 1;
    di.pQueueCreateInfos       = &qci;
    di.enabledExtensionCount   = 1;
    di.ppEnabledExtensionNames = devExts;

    VK_CHECK(vkCreateDevice(physDevice, &di, nullptr, &device));
    vkGetDeviceQueue(device, queueFamily, 0, &queue);
}

void createSwapchain(uint32_t width, uint32_t height) {
    // Query surface capabilities
    VkSurfaceCapabilitiesKHR caps;
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physDevice, surface, &caps));

    // choose extent
    if (caps.currentExtent.width != UINT32_MAX) {
        swapchainExtent = caps.currentExtent;
    } else {
        swapchainExtent = { width, height };
        swapchainExtent.width  = std::clamp(swapchainExtent.width,  caps.minImageExtent.width,  caps.maxImageExtent.width);
        swapchainExtent.height = std::clamp(swapchainExtent.height, caps.minImageExtent.height, caps.maxImageExtent.height);
    }

    // formats
    uint32_t fmtCount=0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physDevice, surface, &fmtCount, nullptr);
    if (!fmtCount) throw std::runtime_error("No surface formats");
    std::vector<VkSurfaceFormatKHR> fmts(fmtCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physDevice, surface, &fmtCount, fmts.data());
    // pick RGBA8 UNORM if possible
    swapchainFormat = fmts[0].format;
    for (auto &f : fmts) {
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM) {
            swapchainFormat = f.format;
            break;
        }
    }

    VkSwapchainCreateInfoKHR sci{};
    sci.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    sci.surface          = surface;
    sci.minImageCount    = caps.minImageCount + 1;
    sci.imageFormat      = swapchainFormat;
    sci.imageColorSpace  = fmts[0].colorSpace;
    sci.imageExtent      = swapchainExtent;
    sci.imageArrayLayers = 1;
    sci.imageUsage       = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sci.preTransform     = caps.currentTransform;
    sci.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sci.presentMode      = VK_PRESENT_MODE_FIFO_KHR;
    sci.clipped          = VK_TRUE;

    VK_CHECK(vkCreateSwapchainKHR(device, &sci, nullptr, &swapchain));

    vkGetSwapchainImagesKHR(device, swapchain, &fmtCount, nullptr);
    swapImages.resize(fmtCount);
    vkGetSwapchainImagesKHR(device, swapchain, &fmtCount, swapImages.data());

    swapImageViews.resize(fmtCount);
    for (uint32_t i = 0; i < fmtCount; i++) {
        VkImageViewCreateInfo ivci{};
        ivci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ivci.image    = swapImages[i];
        ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ivci.format   = swapchainFormat;
        ivci.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        ivci.subresourceRange.levelCount     = 1;
        ivci.subresourceRange.layerCount     = 1;
        VK_CHECK(vkCreateImageView(device, &ivci, nullptr, &swapImageViews[i]));
    }
}

void createStorageImage() {
    storageExtent = swapchainExtent;
    VkImageCreateInfo ici{};
    ici.sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType   = VK_IMAGE_TYPE_2D;
    ici.format      = VK_FORMAT_R8G8B8A8_UNORM;
    ici.extent      = { storageExtent.width, storageExtent.height, 1 };
    ici.mipLevels   = 1;
    ici.arrayLayers = 1;
    ici.usage       = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VK_CHECK(vkCreateImage(device, &ici, nullptr, &storageImage));

    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(device, storageImage, &mr);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(physDevice, &mp);
    VkMemoryAllocateInfo mai{};
    mai.sType          = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((mr.memoryTypeBits & (1<<i)) &&
            (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
           == VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
            mai.memoryTypeIndex = i;
            break;
        }
    }
    VK_CHECK(vkAllocateMemory(device, &mai, nullptr, &storageMemory));
    VK_CHECK(vkBindImageMemory(device, storageImage, storageMemory, 0));

    VkImageViewCreateInfo ivci{};
    ivci.sType                = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivci.image                = storageImage;
    ivci.viewType             = VK_IMAGE_VIEW_TYPE_2D;
    ivci.format               = ici.format;
    ivci.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    ivci.subresourceRange.levelCount     = 1;
    ivci.subresourceRange.layerCount     = 1;
    VK_CHECK(vkCreateImageView(device, &ivci, nullptr, &storageView));
}

void createCameraBuffer() {
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size  = sizeof(Camera);
    bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    VK_CHECK(vkCreateBuffer(device, &bci, nullptr, &cameraBuffer));

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(device, cameraBuffer, &mr);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(physDevice, &mp);

    VkMemoryAllocateInfo mai{};
    mai.sType          = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((mr.memoryTypeBits & (1<<i)) &&
            (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
           == VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
            mai.memoryTypeIndex = i;
            break;
        }
    }
    VK_CHECK(vkAllocateMemory(device, &mai, nullptr, &cameraMemory));
    VK_CHECK(vkBindBufferMemory(device, cameraBuffer, cameraMemory, 0));

    cameraBufferInfo.buffer = cameraBuffer;
    cameraBufferInfo.offset = 0;
    cameraBufferInfo.range  = sizeof(Camera);
}

void createObjectBuffer() {
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size  = sizeof(float) * 16; // two vec4 posRad + two vec4 quat
    bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    VK_CHECK(vkCreateBuffer(device, &bci, nullptr, &objectBuffer));

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(device, objectBuffer, &mr);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(physDevice, &mp);

    VkMemoryAllocateInfo mai{};
    mai.sType          = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((mr.memoryTypeBits & (1<<i)) &&
            (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
           == VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
            mai.memoryTypeIndex = i;
            break;
        }
    }
    VK_CHECK(vkAllocateMemory(device, &mai, nullptr, &objectMemory));
    VK_CHECK(vkBindBufferMemory(device, objectBuffer, objectMemory, 0));

    objectBufferInfo.buffer = objectBuffer;
    objectBufferInfo.offset = 0;
    objectBufferInfo.range  = sizeof(float) * 16;
}

void createDescriptorSet() {
    // storage image binding
    VkDescriptorSetLayoutBinding b0{};  
    b0.binding         = 0;
    b0.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    b0.descriptorCount = 1;
    b0.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    // camera UBO binding
    VkDescriptorSetLayoutBinding b1{};
    b1.binding         = 1;
    b1.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    b1.descriptorCount = 1;
    b1.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    // object UBO binding
    VkDescriptorSetLayoutBinding b2{};
    b2.binding         = 2;
    b2.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    b2.descriptorCount = 1;
    b2.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    std::array<VkDescriptorSetLayoutBinding,3> binds = { b0, b1, b2 };
    VkDescriptorSetLayoutCreateInfo dsli{};
    dsli.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsli.bindingCount = (uint32_t)binds.size();
    dsli.pBindings    = binds.data();
    VK_CHECK(vkCreateDescriptorSetLayout(device, &dsli, nullptr, &dsLayout));

    // pool sizes
    VkDescriptorPoolSize ps0{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1 };
    VkDescriptorPoolSize ps1{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 };
    VkDescriptorPoolSize ps2{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 };
    std::array<VkDescriptorPoolSize,3> pss = { ps0, ps1, ps2 };
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets       = 1;
    dpci.poolSizeCount = (uint32_t)pss.size();
    dpci.pPoolSizes    = pss.data();
    VK_CHECK(vkCreateDescriptorPool(device, &dpci, nullptr, &dsPool));

    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool     = dsPool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &dsLayout;
    VK_CHECK(vkAllocateDescriptorSets(device, &dsai, &ds));

    storageImageInfo.imageView   = storageView;
    storageImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet w0{};
    w0.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w0.dstSet          = ds;
    w0.dstBinding      = 0;
    w0.descriptorCount = 1;
    w0.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w0.pImageInfo      = &storageImageInfo;

    VkWriteDescriptorSet w1{};
    w1.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w1.dstSet          = ds;
    w1.dstBinding      = 1;
    w1.descriptorCount = 1;
    w1.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    w1.pBufferInfo     = &cameraBufferInfo;

    VkWriteDescriptorSet w2{};
    w2.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w2.dstSet          = ds;
    w2.dstBinding      = 2;
    w2.descriptorCount = 1;
    w2.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    w2.pBufferInfo     = &objectBufferInfo;

    std::array<VkWriteDescriptorSet,3> writes = { w0, w1, w2 };
    vkUpdateDescriptorSets(device,
                           (uint32_t)writes.size(), writes.data(),
                            0, nullptr);
}

void createComputePipeline() {
    std::string shaderPath = getExecutableDir() + "/../shaders/comp.spv";
    auto spv = readFile(shaderPath);
    VkShaderModuleCreateInfo smci{};
    smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = spv.size();
    smci.pCode    = reinterpret_cast<const uint32_t*>(spv.data());
    VK_CHECK(vkCreateShaderModule(device, &smci, nullptr, &compShader));

    VkPipelineLayoutCreateInfo plci{};
    plci.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts    = &dsLayout;
    VK_CHECK(vkCreatePipelineLayout(device, &plci, nullptr, &pipelineLayout));

    VkComputePipelineCreateInfo cpci{};
    cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = compShader;
    cpci.stage.pName  = "main";
    cpci.layout       = pipelineLayout;
    VK_CHECK(vkCreateComputePipelines(device,
                                      VK_NULL_HANDLE, 1,
                                      &cpci, nullptr,
                                      &pipeline));
}

void createCommandPoolAndBuffers() {
    VkCommandPoolCreateInfo cpi{};
    cpi.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpi.queueFamilyIndex = queueFamily;
    VK_CHECK(vkCreateCommandPool(device, &cpi, nullptr, &cmdPool));

    uint32_t count = headless ? 1 : (uint32_t)swapImageViews.size();
    cmdBuffers.resize(count);
    VkCommandBufferAllocateInfo cbai{};
    cbai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool        = cmdPool;
    cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = count;
    VK_CHECK(vkAllocateCommandBuffers(device, &cbai, cmdBuffers.data()));
}

void createSyncObjects() {
    if(headless) return;
    VkSemaphoreCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VK_CHECK(vkCreateSemaphore(device, &sci, nullptr, &semImageAvailable));
    VK_CHECK(vkCreateSemaphore(device, &sci, nullptr, &semRenderFinished));
}

void cleanupSwapchain() {
    for (auto view : swapImageViews)
        vkDestroyImageView(device, view, nullptr);
    swapImageViews.clear();
    if (swapchain)
        vkDestroySwapchainKHR(device, swapchain, nullptr);

    if (storageView)
        vkDestroyImageView(device, storageView, nullptr);
    if (storageImage)
        vkDestroyImage(device, storageImage, nullptr);
    if (storageMemory)
        vkFreeMemory(device, storageMemory, nullptr);

    if (dsPool)
        vkDestroyDescriptorPool(device, dsPool, nullptr);
    if (cmdPool)
        vkDestroyCommandPool(device, cmdPool, nullptr);
    if(headless)
        destroyHeadlessBuffer();
}

void recreateSwapchain(uint32_t width, uint32_t height) {
    vkDeviceWaitIdle(device);
    cleanupSwapchain();
    createSwapchain(width, height);
    createStorageImage();
    createDescriptorSet();
    createCommandPoolAndBuffers();
}

void createHeadlessBuffer() {
    if(!headless) return;
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size  = storageExtent.width * storageExtent.height * 4;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VK_CHECK(vkCreateBuffer(device, &bci, nullptr, &headlessBuffer));

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(device, headlessBuffer, &mr);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(physDevice, &mp);
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    for(uint32_t i=0;i<mp.memoryTypeCount;i++){
        if((mr.memoryTypeBits & (1u<<i)) &&
           (mp.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
           (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)){
            mai.memoryTypeIndex = i;
            break;
        }
    }
    VK_CHECK(vkAllocateMemory(device, &mai, nullptr, &headlessMemory));
    VK_CHECK(vkBindBufferMemory(device, headlessBuffer, headlessMemory, 0));
}

void destroyHeadlessBuffer(){
    if(headlessBuffer)
        vkDestroyBuffer(device, headlessBuffer, nullptr);
    if(headlessMemory)
        vkFreeMemory(device, headlessMemory, nullptr);
    headlessBuffer = VK_NULL_HANDLE;
    headlessMemory = VK_NULL_HANDLE;
}

static void savePPM(const char* path, uint32_t w, uint32_t h, const void* data){
    std::ofstream f(path, std::ios::binary);
    f<<"P6\n"<<w<<" "<<h<<"\n255\n";
    const uint8_t* px = reinterpret_cast<const uint8_t*>(data);
    for(uint32_t i=0;i<w*h;i++){
        f.put(px[i*4+0]);
        f.put(px[i*4+1]);
        f.put(px[i*4+2]);
    }
}

void saveScreenshot(const char* path, Camera& cam, const FractalObject& a, const FractalObject& b){
    // allocate temporary buffer
    VkBuffer buf;
    VkDeviceMemory mem;
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = storageExtent.width * storageExtent.height * 4;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VK_CHECK(vkCreateBuffer(device, &bci, nullptr, &buf));

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(device, buf, &mr);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(physDevice, &mp);
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    for(uint32_t i=0;i<mp.memoryTypeCount;i++){
        if((mr.memoryTypeBits & (1u<<i)) &&
           (mp.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)){
            mai.memoryTypeIndex = i;
            break;
        }
    }
    VK_CHECK(vkAllocateMemory(device, &mai, nullptr, &mem));
    VK_CHECK(vkBindBufferMemory(device, buf, mem, 0));

    // update UBOs
    void* ptr;
    vkMapMemory(device, cameraMemory, 0, sizeof(cam), 0, &ptr);
    std::memcpy(ptr, &cam, sizeof(cam));
    vkUnmapMemory(device, cameraMemory);

    struct { float posRad[2][4]; float quat[2][4]; } odata;
    odata.posRad[0][0]=a.position.x; odata.posRad[0][1]=a.position.y; odata.posRad[0][2]=a.position.z; odata.posRad[0][3]=a.radius;
    odata.posRad[1][0]=b.position.x; odata.posRad[1][1]=b.position.y; odata.posRad[1][2]=b.position.z; odata.posRad[1][3]=b.radius;
    odata.quat[0][0]=a.orientation.x; odata.quat[0][1]=a.orientation.y; odata.quat[0][2]=a.orientation.z; odata.quat[0][3]=a.orientation.w;
    odata.quat[1][0]=b.orientation.x; odata.quat[1][1]=b.orientation.y; odata.quat[1][2]=b.orientation.z; odata.quat[1][3]=b.orientation.w;
    vkMapMemory(device, objectMemory, 0, sizeof(odata), 0, &ptr);
    std::memcpy(ptr, &odata, sizeof(odata));
    vkUnmapMemory(device, objectMemory);

    // command buffer
    VkCommandBuffer cb;
    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = cmdPool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(device, &cbai, &cb));

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CHECK(vkBeginCommandBuffer(cb, &bi));

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.image = storageImage;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,0,nullptr,0,nullptr,1,&barrier);

    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout,0,1,&ds,0,nullptr);
    vkCmdDispatch(cb, (storageExtent.width+15)/16, (storageExtent.height+15)/16, 1);

    barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,0,nullptr,0,nullptr,1,&barrier);

    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {storageExtent.width, storageExtent.height,1};
    vkCmdCopyImageToBuffer(cb, storageImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buf, 1, &region);

    VkBufferMemoryBarrier bmb{};
    bmb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bmb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bmb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    bmb.buffer = buf;
    bmb.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,0,nullptr,1,&bmb,0,nullptr);

    VK_CHECK(vkEndCommandBuffer(cb));

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
    vkQueueWaitIdle(queue);

    vkMapMemory(device, mem, 0, VK_WHOLE_SIZE, 0, &ptr);
    savePPM(path, storageExtent.width, storageExtent.height, ptr);
    vkUnmapMemory(device, mem);

    vkFreeCommandBuffers(device, cmdPool, 1, &cb);
    vkDestroyBuffer(device, buf, nullptr);
    vkFreeMemory(device, mem, nullptr);
}

void drawFrameHeadless(uint32_t frame, Camera& cam, const FractalObject& a, const FractalObject& b){
    if(!headless) return;
    // update camera UBO
    void* ptr;
    vkMapMemory(device, cameraMemory, 0, sizeof(cam), 0, &ptr);
    std::memcpy(ptr, &cam, sizeof(cam));
    vkUnmapMemory(device, cameraMemory);

    struct {
        float posRad[2][4];
        float quat[2][4];
    } odata;
    odata.posRad[0][0] = a.position.x; odata.posRad[0][1] = a.position.y; odata.posRad[0][2] = a.position.z; odata.posRad[0][3] = a.radius;
    odata.posRad[1][0] = b.position.x; odata.posRad[1][1] = b.position.y; odata.posRad[1][2] = b.position.z; odata.posRad[1][3] = b.radius;
    odata.quat[0][0] = a.orientation.x; odata.quat[0][1] = a.orientation.y; odata.quat[0][2] = a.orientation.z; odata.quat[0][3] = a.orientation.w;
    odata.quat[1][0] = b.orientation.x; odata.quat[1][1] = b.orientation.y; odata.quat[1][2] = b.orientation.z; odata.quat[1][3] = b.orientation.w;
    vkMapMemory(device, objectMemory, 0, sizeof(odata), 0, &ptr);
    std::memcpy(ptr, &odata, sizeof(odata));
    vkUnmapMemory(device, objectMemory);

    VkCommandBuffer cb = cmdBuffers[0];
    vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CHECK(vkBeginCommandBuffer(cb, &bi));

    // image for compute
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.image = storageImage;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,0,nullptr,0,nullptr,1,&barrier);

    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0,1,&ds,0,nullptr);
    vkCmdDispatch(cb, (storageExtent.width+15)/16, (storageExtent.height+15)/16, 1);

    // transition for copy
    barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,0,nullptr,0,nullptr,1,&barrier);

    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {storageExtent.width, storageExtent.height,1};
    vkCmdCopyImageToBuffer(cb, storageImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, headlessBuffer, 1, &region);

    VkBufferMemoryBarrier bmb{};
    bmb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bmb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bmb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    bmb.buffer = headlessBuffer;
    bmb.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,0,nullptr,1,&bmb,0,nullptr);

    VK_CHECK(vkEndCommandBuffer(cb));

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
    vkQueueWaitIdle(queue);

    vkMapMemory(device, headlessMemory, 0, VK_WHOLE_SIZE, 0, &ptr);
    char name[64];
    snprintf(name, sizeof(name), "frame_%04u.ppm", frame);
    savePPM(name, storageExtent.width, storageExtent.height, ptr);
    vkUnmapMemory(device, headlessMemory);
}

// One‐time record & submit per frame:
void drawFrame(uint32_t /*unused*/, Camera &cam, const FractalObject& a, const FractalObject& b) {
    // acquire
    uint32_t imageIndex;
    VK_CHECK(vkAcquireNextImageKHR(device, swapchain,
        UINT64_MAX, semImageAvailable, VK_NULL_HANDLE,
        &imageIndex));

    // update camera UBO
    void* ptr;
    vkMapMemory(device, cameraMemory, 0,
                sizeof(cam), 0, &ptr);
    std::memcpy(ptr, &cam, sizeof(cam));
    vkUnmapMemory(device, cameraMemory);

    struct {
        float posRad[2][4];
        float quat[2][4];
    } odata;
    odata.posRad[0][0] = a.position.x; odata.posRad[0][1] = a.position.y; odata.posRad[0][2] = a.position.z; odata.posRad[0][3] = a.radius;
    odata.posRad[1][0] = b.position.x; odata.posRad[1][1] = b.position.y; odata.posRad[1][2] = b.position.z; odata.posRad[1][3] = b.radius;
    odata.quat[0][0] = a.orientation.x; odata.quat[0][1] = a.orientation.y; odata.quat[0][2] = a.orientation.z; odata.quat[0][3] = a.orientation.w;
    odata.quat[1][0] = b.orientation.x; odata.quat[1][1] = b.orientation.y; odata.quat[1][2] = b.orientation.z; odata.quat[1][3] = b.orientation.w;
    vkMapMemory(device, objectMemory, 0, sizeof(odata), 0, &ptr);
    std::memcpy(ptr, &odata, sizeof(odata));
    vkUnmapMemory(device, objectMemory);

    // record
    VkCommandBuffer cb = cmdBuffers[imageIndex];
    vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CHECK(vkBeginCommandBuffer(cb, &bi));

    // transition storageImage -> GENERAL for compute
    {
        VkImageMemoryBarrier barrier{};
        barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
        barrier.image               = storageImage;
        barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.levelCount     = 1;
        barrier.subresourceRange.layerCount     = 1;
        barrier.srcAccessMask       = 0;
        barrier.dstAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;

        vkCmdPipelineBarrier(cb,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0,nullptr, 0,nullptr,
            1,&barrier);
    }

    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
        pipelineLayout, 0, 1, &ds, 0, nullptr);

    vkCmdDispatch(cb,
        (storageExtent.width  +15)/16,
        (storageExtent.height +15)/16,
        1);

    // transition storageImage -> TRANSFER_SRC
    {
        VkImageMemoryBarrier barrier{};
        barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout           = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.image               = storageImage;
        barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.levelCount     = 1;
        barrier.subresourceRange.layerCount     = 1;
        barrier.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;

        vkCmdPipelineBarrier(cb,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,0,nullptr,0,nullptr,
            1,&barrier);
    }

    // transition swapchain image -> TRANSFER_DST
    {
        VkImageMemoryBarrier barrier{};
        barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.image               = swapImages[imageIndex];
        barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.levelCount     = 1;
        barrier.subresourceRange.layerCount     = 1;
        barrier.srcAccessMask       = 0;
        barrier.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;

        vkCmdPipelineBarrier(cb,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,0,nullptr,0,nullptr,
            1,&barrier);
    }

    // copy storage -> swapchain
    {
        VkImageCopy copyRegion{};
        // source subresource
        copyRegion.srcSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.srcSubresource.mipLevel       = 0;
        copyRegion.srcSubresource.baseArrayLayer = 0;
        copyRegion.srcSubresource.layerCount     = 1;
        copyRegion.srcOffset                     = {0,0,0};

        // destination subresource
        copyRegion.dstSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.dstSubresource.mipLevel       = 0;
        copyRegion.dstSubresource.baseArrayLayer = 0;
        copyRegion.dstSubresource.layerCount     = 1;
        copyRegion.dstOffset                     = {0,0,0};

        // full extent copy
        copyRegion.extent = {
            swapchainExtent.width,
            swapchainExtent.height,
            1
        };

        vkCmdCopyImage(cb,
            storageImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            swapImages[imageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &copyRegion);
    }

    // transition swapchain -> PRESENT_SRC
    {
        VkImageMemoryBarrier barrier{};
        barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout           = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        barrier.image               = swapImages[imageIndex];
        barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.levelCount     = 1;
        barrier.subresourceRange.layerCount     = 1;
        barrier.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask       = VK_ACCESS_MEMORY_READ_BIT;

        vkCmdPipelineBarrier(cb,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0,0,nullptr,0,nullptr,
            1,&barrier);
    }

    VK_CHECK(vkEndCommandBuffer(cb));

    // submit
    VkSubmitInfo si{};
    si.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.waitSemaphoreCount   = 1;
    si.pWaitSemaphores      = &semImageAvailable;
    VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_TRANSFER_BIT };
    si.pWaitDstStageMask    = waitStages;
    si.commandBufferCount   = 1;
    si.pCommandBuffers      = &cb;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores    = &semRenderFinished;

    VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));

    // present
    VkPresentInfoKHR pi{};
    pi.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores    = &semRenderFinished;
    pi.swapchainCount     = 1;
    pi.pSwapchains        = &swapchain;
    pi.pImageIndices      = &imageIndex;

    VK_CHECK(vkQueuePresentKHR(queue, &pi));
    vkQueueWaitIdle(queue);
}

int main() {
    try {
        createInstance();
        createWindowAndSurface();
        pickPhysicalDevice();
        createLogicalDeviceAndQueue();
        int fbw = WIDTH, fbh = HEIGHT;
        if(!headless){
            glfwGetFramebufferSize(window, &fbw, &fbh);
            createSwapchain(fbw, fbh);
        } else {
            swapchainExtent = { (uint32_t)fbw, (uint32_t)fbh };
        }
        createStorageImage();
        createCameraBuffer();
        createObjectBuffer();
        createDescriptorSet();
        createComputePipeline();
        createCommandPoolAndBuffers();
        if(headless)
            createHeadlessBuffer();
        else
            createSyncObjects();

        // initial camera
        Camera cam{};
        cam.pos[0]=0; cam.pos[1]=0; cam.pos[2]=3;

        Quat camRot{1.f,0.f,0.f,0.f};
        rotateVec(camRot, BASE_FORWARD, cam.forward);
        rotateVec(camRot, BASE_UP,      cam.up);
        rotateVec(camRot, BASE_RIGHT,   cam.right);

        float fracRad = estimateSierpinskiRadius() * 50.f;
        FractalObject objA{
            {-2.f,0.f,0.f}, // position
            {0.f,0.f,0.f},  // velocity
            {0.f,0.5f,0.f},  // angular velocity
            {1.f,0.f,0.f,0.f}, // orientation
            fracRad,
            1.f,  // mass
            0.4f, // inertia
            sierpinskiDE
        };
        FractalObject objB{
            {2.f,0.f,0.f},
            {0.f,0.f,0.f},
            {0.f,-0.5f,0.f},
            {1.f,0.f,0.f,0.f},
            fracRad,
            1.f,
            0.4f,
            sierpinskiDE
        };

        double lastX = WIDTH/2.0, lastY = HEIGHT/2.0;
        if(!headless){
            glfwSetCursorPos(window, lastX, lastY);
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        }

        auto lastTime = now();
        while (!headless && !glfwWindowShouldClose(window)) {
            glfwPollEvents();
            int curW, curH;
            glfwGetFramebufferSize(window, &curW, &curH);
            if (curW != (int)swapchainExtent.width || curH != (int)swapchainExtent.height)
                recreateSwapchain(curW, curH);
            // delta
            auto t2 = now();
            float dt = std::chrono::duration<float>(t2 - lastTime).count();
            lastTime = t2;

            stepPhysics(objA, objB, dt);

            // mouse look
            double mx,my;
            glfwGetCursorPos(window, &mx, &my);
            float dx = float(mx - lastX), dy = float(my - lastY);
            lastX = mx; lastY = my;

            const float sens = 0.0025f;
            float yaw   = -dx * sens;
            float pitch = -dy * sens;

            float upAxis[3];
            rotateVec(camRot, BASE_UP, upAxis);
            if(yaw != 0.f) {
                camRot = quatMul(quatFromAxisAngle(upAxis, yaw), camRot);
            }
            float rightAxis[3];
            rotateVec(camRot, BASE_RIGHT, rightAxis);
            if(pitch != 0.f) {
                camRot = quatMul(quatFromAxisAngle(rightAxis, pitch), camRot);
            }

            float roll = 0.f;
            if(glfwGetKey(window, GLFW_KEY_Q)==GLFW_PRESS) roll += 1.f;
            if(glfwGetKey(window, GLFW_KEY_E)==GLFW_PRESS) roll -= 1.f;
            if(roll != 0.f) {
                float fwdAxis[3];
                rotateVec(camRot, BASE_FORWARD, fwdAxis);
                camRot = quatMul(quatFromAxisAngle(fwdAxis, roll*1.5f*dt), camRot);
            }

            quatNormalize(camRot);
            rotateVec(camRot, BASE_FORWARD, cam.forward);
            rotateVec(camRot, BASE_UP,      cam.up);
            rotateVec(camRot, BASE_RIGHT,   cam.right);

            float mvF=0.f,mvR=0.f,mvU=0.f;
            if(glfwGetKey(window, GLFW_KEY_W)==GLFW_PRESS) mvF += 1.f;
            if(glfwGetKey(window, GLFW_KEY_S)==GLFW_PRESS) mvF -= 1.f;
            if(glfwGetKey(window, GLFW_KEY_D)==GLFW_PRESS) mvR += 1.f;
            if(glfwGetKey(window, GLFW_KEY_A)==GLFW_PRESS) mvR -= 1.f;
            if(glfwGetKey(window, GLFW_KEY_SPACE)==GLFW_PRESS) mvU += 1.f;
            if(glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
               glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS) mvU -= 1.f;

            float move[3] = {
                cam.forward[0]*mvF + cam.right[0]*mvR + cam.up[0]*mvU,
                cam.forward[1]*mvF + cam.right[1]*mvR + cam.up[1]*mvU,
                cam.forward[2]*mvF + cam.right[2]*mvR + cam.up[2]*mvU
            };
            float mlen = std::sqrt(move[0]*move[0]+move[1]*move[1]+move[2]*move[2]);
            if(mlen>0.f){ move[0]/=mlen; move[1]/=mlen; move[2]/=mlen; }
            const float speed=3.0f;
            cam.pos[0] += move[0]*speed*dt;
            cam.pos[1] += move[1]*speed*dt;
            cam.pos[2] += move[2]*speed*dt;

            drawFrame(0, cam, objA, objB);
            if(screenshotPending){
                char name[64];
                snprintf(name, sizeof(name), "screenshot_%04u.ppm", screenshotIndex++);
                saveScreenshot(name, cam, objA, objB);
                screenshotPending = false;
            }
        }
        if(headless){
            for(uint32_t i=0;i<60;i++){
                stepPhysics(objA, objB, 0.016f);
                float t = i*0.05f;
                cam.pos[0] = 3.0f*std::cos(t);
                cam.pos[1] = 0.0f;
                cam.pos[2] = 3.0f*std::sin(t);
                Vec3 fwd = normalize(Vec3{-cam.pos[0], -cam.pos[1], -cam.pos[2]});
                Vec3 right = normalize(cross(fwd, Vec3{0.f,1.f,0.f}));
                Vec3 upv = cross(right, fwd);
                cam.forward[0]=fwd.x; cam.forward[1]=fwd.y; cam.forward[2]=fwd.z;
                cam.right[0]=right.x; cam.right[1]=right.y; cam.right[2]=right.z;
                cam.up[0]=upv.x; cam.up[1]=upv.y; cam.up[2]=upv.z;
                drawFrameHeadless(i, cam, objA, objB);
            }
        }

        vkDeviceWaitIdle(device);
        // TODO: cleanup all Vulkan resources...
    }
    catch (std::exception &e) {
        std::cerr<<"Fatal: "<<e.what()<<"\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
