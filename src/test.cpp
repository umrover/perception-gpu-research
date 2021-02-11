#include <vector>
#include <mutex>
#include <iostream>
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cuda.h>
#include <cuda_gl_interop.h>

void checkError(cudaError_t err) {
    if(err != cudaSuccess)
        std::cerr << "Error: (" << err << "): " << cudaGetErrorString(err) << std::endl;
}

int main(int argc, char **argv) {

    GLuint bufferGLID_;
    cudaGraphicsResource* bufferCudaID_;

    glutInit(&argc, argv);
    int wnd_w = glutGet(GLUT_SCREEN_WIDTH);
    int wnd_h = glutGet(GLUT_SCREEN_HEIGHT) *0.9;
    glutInitWindowSize(wnd_w*0.9, wnd_h*0.9);
    glutInitWindowPosition(wnd_w*0.05, wnd_h*0.05);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutCreateWindow("ZED Depth Sensing");

    GLenum err = glewInit();
    if (GLEW_OK != err)
        return err;

    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);
    glEnable(GL_DEPTH_TEST);

    glGenBuffers(1, &bufferGLID_);
    glBindBuffer(GL_ARRAY_BUFFER, bufferGLID_);
    glBufferData(GL_ARRAY_BUFFER, 14400 * 4 * sizeof(float), 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    checkError(cudaGraphicsGLRegisterBuffer(&bufferCudaID_, bufferGLID_, cudaGraphicsRegisterFlagsNone));

    while(true){}
}