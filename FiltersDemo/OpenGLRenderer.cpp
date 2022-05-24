#include "OpenGLRenderer.h"
#include <fstream>

OpenGLRenderer::OpenGLRenderer(int windowWidth, int windowHeight, OpenGLWrapper* _openglwrapper, ImGuiMenu* _menu, Image* _image) : openglwrapper(_openglwrapper)

{
	width = windowWidth;
	height = windowHeight;
	openglwrapper->MakeArrayAndVertexBuffers();
	openglwrapper->BindVAO();
	openglwrapper->BindVBO();
	float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
	openglwrapper->LoadDataToBuffers(sizeof(vertexCoords), vertexCoords);
	location = openglwrapper->GetUniformLocation("textureUnit");
	image = _image;
}

void OpenGLRenderer::UpdateTexture()
{
	openglwrapper->DefineTexture(width, height, image->GetData());
}

void OpenGLRenderer::Draw()
{
	openglwrapper->BindVAO();

	if (location >= 0)
	{
		openglwrapper->SetUniform1i(location);
		openglwrapper->BindTexture();
	}

	openglwrapper->Draw();	// draw two triangles forming a quad
}

Image* OpenGLRenderer::GetImage()
{
	return image;
}


OpenGLRenderer::~OpenGLRenderer()
{
	delete image;
	delete openglwrapper;
}

void OpenGLRenderer::SetMenu(ImGuiMenu* _menu)
{
	menu = _menu;
}