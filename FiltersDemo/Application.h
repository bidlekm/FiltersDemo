#pragma once
#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "OpenGLRenderer.h"
#include "ImGuiMenu.h"
#include "OpenGLWrapper.h"
#include "Image.h"

class Application
{
	GLFWwindow* window;
	Image* image;
	OpenGLWrapper* openglwrapper;
	OpenGLRenderer* openGlRenderer;
	ImGuiMenu* menu;
public:
	Application();
	void Run();
	~Application();
};

