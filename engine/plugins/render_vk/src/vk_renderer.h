#pragma once

#include <vulkan/vulkan.h>

#include <unordered_map>
#include <vector>

#include "orange/render/renderer.h"

struct SDL_Window;

namespace orange::vk {

// A GPU buffer object: a VkBuffer backed by host-visible device memory.
struct VkBufferObject {
    VkBuffer       buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize   size   = 0;
};

// A GPU texture: image + memory + view + its descriptor set.
struct VkTexture {
    VkImage         image  = VK_NULL_HANDLE;
    VkDeviceMemory  memory = VK_NULL_HANDLE;
    VkImageView     view   = VK_NULL_HANDLE;
    VkDescriptorSet set    = VK_NULL_HANDLE;
};

// A mesh references buffer objects (it does not own them) + draw metadata.
struct VkMesh {
    render::BufferHandle vertexBuffer = render::kInvalidBuffer;
    render::BufferHandle indexBuffer  = render::kInvalidBuffer;
    uint32_t             vertexCount  = 0;
    uint32_t             indexCount   = 0;
    bool                 indexed      = false;
    bool                 points       = false;  // point cloud -> sphere-imposter points
};

// Minimal Vulkan backend: brings up instance/device/swapchain and clears +
// presents each frame through a real render pass. Mesh upload/draw is the
// documented next step (see submit()); the plugin boundary and frame loop are
// already production-shaped.
class VkRenderer final : public render::IRenderer {
public:
    bool init(const render::InitInfo& info) override;
    void shutdown() override;

    render::BufferHandle createBuffer(const render::BufferDesc& desc) override;
    void                 updateBuffer(render::BufferHandle buffer, const void* data,
                                      size_t size, size_t offset) override;
    void                 destroyBuffer(render::BufferHandle buffer) override;

    render::TextureHandle createTexture(const render::TextureDesc& desc) override;
    void                  destroyTexture(render::TextureHandle texture) override;

    render::MeshHandle createMesh(const render::MeshDesc& desc) override;
    void               destroyMesh(render::MeshHandle mesh) override;

    void beginFrame(const render::FrameContext& frame) override;
    void submit(const render::DrawItem& item) override;
    void beginOverlay(const render::OverlayContext& overlay) override;
    void endFrame() override;
    void drawGrid(int upAxis) override;

    void resize(uint32_t width, uint32_t height) override;
    void setVsync(bool enabled) override;
    void setDrawMode(uint32_t mode) override;
    void setPointSize(float pixels) override;
    void setLighting(bool enabled) override;
    void setCrossSection(bool enabled, const float plane[4]) override;
    void setColorMode(uint32_t mode) override;
    bool readPixels(std::vector<uint8_t>& out, uint32_t& w, uint32_t& h) override;

    render::Backend backend() const override { return render::Backend::Vulkan; }
    const char*     name() const override { return "Orange Vulkan Renderer"; }

private:
    static constexpr int kFramesInFlight = 2;

    SDL_Window*      window_ = nullptr;
    VkInstance       instance_ = VK_NULL_HANDLE;
    VkSurfaceKHR     surface_  = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice         device_   = VK_NULL_HANDLE;
    uint32_t         queueFamily_ = 0;
    VkQueue          queue_    = VK_NULL_HANDLE;

    VkSwapchainKHR             swapchain_ = VK_NULL_HANDLE;
    VkFormat                  swapFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D                swapExtent_{0, 0};
    std::vector<VkImage>      swapImages_;
    std::vector<VkImageView>  swapViews_;
    std::vector<VkFramebuffer> framebuffers_;
    VkRenderPass              renderPass_ = VK_NULL_HANDLE;

    // Depth buffer (one, recreated with the swapchain).
    VkImage        depthImage_  = VK_NULL_HANDLE;
    VkDeviceMemory depthMemory_ = VK_NULL_HANDLE;
    VkImageView    depthView_   = VK_NULL_HANDLE;
    VkFormat       depthFormat_ = VK_FORMAT_D24_UNORM_S8_UINT;  // depth + stencil (outline)

    // Graphics pipeline for mesh rendering (built lazily from the first mesh's
    // vertex layout; all appOrange meshes share the same layout).
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline       pipeline_       = VK_NULL_HANDLE;  // solid (fill, slight depth bias)
    VkPipeline       pipelineWireframe_ = VK_NULL_HANDLE;  // line polygon mode
    VkPipeline       pipelinePoint_     = VK_NULL_HANDLE;  // point polygon mode
    VkPipeline       pipelineOutline_   = VK_NULL_HANDLE;  // enlarged, stencil!=1 (silhouette)
    VkPipeline       pipelineStencilMask_ = VK_NULL_HANDLE;  // solid, writes stencil ref 1 only
    VkPipeline       pipelinePoints_      = VK_NULL_HANDLE;  // POINT_LIST sphere imposters
    uint32_t         drawMode_ = 1;  // Helium DrawingMode (0=none..4=point)
    float            pointSize_ = 6.0f;  // point-cloud sprite pixel diameter
    bool             lighting_  = true;  // point-sprite diffuse shading on/off (`)
    float            clipPlane_[4] = {0, 0, 0, 0};  // cross-section (nx,ny,nz,d); 0 = off
    uint32_t         colorMode_ = 0;     // scene coloring mode (Shift+` cycles)
    bool             inOverlay_ = false;  // overlay passes are never cross-sectioned/recolored

    // Infinite-grid pipeline: a vertex-less full-screen pass (own layout with a
    // 128-byte push constant = viewProj + invViewProj, vertex + fragment stages).
    VkPipelineLayout gridLayout_   = VK_NULL_HANDLE;
    VkPipeline       gridPipeline_ = VK_NULL_HANDLE;

    VkCommandPool                commandPool_ = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers_;
    std::vector<VkSemaphore>     imageAvailable_;
    std::vector<VkSemaphore>     renderFinished_;
    std::vector<VkFence>         inFlight_;

    uint32_t currentFrame_ = 0;
    uint32_t imageIndex_   = 0;
    bool     frameActive_  = false;
    bool     needsResize_  = false;
    bool     vsync_        = true;  // FIFO present mode when set
    uint32_t pendingW_ = 0, pendingH_ = 0;
    float    clear_[4] = {0.05f, 0.06f, 0.08f, 1.0f};

    render::BufferHandle nextBuffer_ = 1;
    std::unordered_map<render::BufferHandle, VkBufferObject> buffers_;

    render::MeshHandle nextMesh_ = 1;
    std::unordered_map<render::MeshHandle, VkMesh> meshes_;

    // Texturing: sampler + descriptor infrastructure + a 1x1 white fallback.
    VkSampler             sampler_     = VK_NULL_HANDLE;
    VkDescriptorSetLayout descLayout_  = VK_NULL_HANDLE;
    VkDescriptorPool      descPool_    = VK_NULL_HANDLE;
    render::TextureHandle nextTexture_ = 1;
    std::unordered_map<render::TextureHandle, VkTexture> textures_;
    VkDescriptorSet       whiteSet_    = VK_NULL_HANDLE;

    // Per-frame view*proj (with Vulkan clip correction), pushed each draw, plus
    // its inverse for the grid's screen-ray unprojection.
    float viewProj_[16];
    float invViewProj_[16];

    uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) const;
    bool     createTextureResources();  // sampler, descriptor layout/pool, white tex
    bool     createGridPipeline();      // vertex-less infinite-grid pipeline

    bool createInstance();
    bool createSurface();
    bool pickPhysicalDevice();
    bool createDevice();
    bool createSwapchain();
    bool createRenderPass();
    bool createFramebuffers();
    bool createCommands();
    bool createSyncObjects();
    bool createDepthResources();
    bool createPipeline(const render::VertexLayout& layout);

    void destroyDepthResources();
    void destroySwapchain();
    bool recreateSwapchain();
};

} // namespace orange::vk
