# Metharizon

This is a small sample application that uses Vulkan compute shaders to raymarch a pair of Sierpiński tetrahedra. It can run interactively in a window or in headless mode when no display is available.

## Building

```bash
cmake -S . -B build
cmake --build build
```

The SPIR-V compute shader will be compiled automatically during the build.

## Running

Execute the generated binary:

```bash
./build/Metharizon
```

Press **F11** to toggle fullscreen. Press **ESC** to quit.

## Screenshots

While running interactively you can press **P** to capture the current frame. Screenshots are written as `screenshot_XXXX.ppm` in the working directory.


