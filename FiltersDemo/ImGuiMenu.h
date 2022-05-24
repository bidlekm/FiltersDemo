#pragma once
#include <string>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "glm.hpp"
#include "OpenGLRenderer.h"
#include <chrono>




struct ImGuiMenu
{
	static int ID;
	GLFWwindow* window;
	OpenGLRenderer* openGlRenderer;
	const char* noiseTypes[1] = { "Gaussian White" };
	const char* denoiseAlgorithms[2] = { "Gradient Descent", "Primal Dual" };
	const char* currentNoiseType, *currentDenoiseAlgorithm;

	ImGuiMenu(GLFWwindow* _window,OpenGLRenderer* _openGlRenderer);

	void Draw() ;

	void Update() ;

	void Theme();

	void Kill() ;

	float AddFloat(float value, float min, float max);

	glm::vec3 AddVec3(glm::vec3 value, float min, float max);

	glm::vec2 AddVec2(glm::vec2 value, float min, float max);

	glm::vec3 AddColor3f(glm::vec3 value);

	void SetOpenGLRenderer(OpenGLRenderer* _openglrenderer);

	~ImGuiMenu();
};

