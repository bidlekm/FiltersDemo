#pragma once 

#include "OpenGLWrapper.h"
#include "glm.hpp"
#include "Image.h"

struct ImGuiMenu;

class OpenGLRenderer {
private:
	int height, width;
	ImGuiMenu* menu;
	unsigned int vao;
	int location;
	OpenGLWrapper* openglwrapper;
	Image* image;

public:
	OpenGLRenderer(int windowWidth, int windowHeight, OpenGLWrapper* _openglwrapper, ImGuiMenu* _menu, Image* image);

	void UpdateTexture();

	void Draw();

	Image* GetImage();

	~OpenGLRenderer();

	void SetMenu(ImGuiMenu* menu);

};

