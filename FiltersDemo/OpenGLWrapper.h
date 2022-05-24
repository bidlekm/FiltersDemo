#pragma once
#include <GL/glew.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>

#define ASSERT(x) if(!(x)) __debugbreak();
#define GLCall(x) GLClearError();\
	x;\
	ASSERT(GLLogCall(#x, __FILE__, __LINE__))


void GLClearError();
bool GLLogCall(const char* function, const char* file, int line);


struct ShaderSource
{
	std::string VertexSource;
	std::string FragmentSource;
	std::string ComputeSource;
};

class OpenGLWrapper
{
private:
	std::string m_ShaderFilePath;
	unsigned int m_ShaderProgramID;
	unsigned int m_VBO;
	unsigned int m_VAO;
	unsigned int m_TextureID;
	std::unordered_map<std::string, int> m_UniformLocationCache;
public:
	OpenGLWrapper(const std::string& filepath);
	~OpenGLWrapper();

	unsigned int CompileShader(const std::string& source, unsigned int type);
	ShaderSource ParseShader(const std::string& filePath);
	unsigned int CreateShader(const std::string& vs, const std::string& fs);

	void MakeShaderProgram();
	void BindShaderProgram() const;
	void UnBindShaderProgram() const;

	void MakeTexture();

	void MakeArrayAndVertexBuffers();
	void LoadDataToBuffers(GLsizeiptr size, const void* data);
	void BindVAO() const;
	void UnBindVAO() const;

	void BindVBO() const;
	void UnBindVBO() const;

	void SetUniform1i(int location);

	int GetUniformLocation(const std::string& name);

	void printInfo();

	void Draw();

	unsigned int GetRenderID() const;

	unsigned int GetTextureID() const;

	void DefineTexture(int width, int height, const void* data);

	void BindTexture() const;

	void DeleteTexture();
};

