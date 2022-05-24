#include "OpenGLWrapper.h"

void GLClearError()
{

	while (glGetError() != GL_NO_ERROR);

}
bool GLLogCall(const char* function, const char* file, int line)
{

	while (GLenum error = glGetError())
	{
		std::cout << "[OpenGL Error] (" << std::hex << error << ") in " << function
			<< " " << file << ": (" << std::dec << line << ")" << std::endl;
		return false;
	}
	return true;
}


OpenGLWrapper::OpenGLWrapper(const std::string& filepath) : m_ShaderFilePath(filepath), m_ShaderProgramID(0),
m_TextureID(0)
{
	MakeShaderProgram();
	MakeTexture();
	MakeArrayAndVertexBuffers();
}

void OpenGLWrapper::MakeShaderProgram() {

	ShaderSource source = ParseShader(m_ShaderFilePath);
	m_ShaderProgramID = CreateShader(source.VertexSource, source.FragmentSource);
}

void OpenGLWrapper::MakeTexture()
{
	GLCall(glGenTextures(1, &m_TextureID));  	// id generation
	GLCall(glBindTexture(GL_TEXTURE_2D, m_TextureID));    // binding
	GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)); // sampling
	GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
}

void OpenGLWrapper::MakeArrayAndVertexBuffers()
{
	GLCall(glGenVertexArrays(1, &m_VAO));	// create 1 vertex array object
	GLCall(glGenBuffers(1,  &m_VBO));	// Generate 1 vertex buffer objects
}

void OpenGLWrapper::LoadDataToBuffers(GLsizeiptr size, const void* data) {
	GLCall(glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW));	   // copy to that part of the memory which is not modified 
	GLCall(glEnableVertexAttribArray(0));
	GLCall(glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL));     // stride and offset: it is tightly packed

}

void OpenGLWrapper::BindVAO() const
{
	GLCall(glBindVertexArray(m_VAO));
}

void OpenGLWrapper::UnBindVAO() const
{
	GLCall(glBindVertexArray(0));
}

void OpenGLWrapper::BindVBO() const
{
	GLCall(glBindBuffer(GL_ARRAY_BUFFER, m_VBO));
}

void OpenGLWrapper::UnBindVBO() const
{
	GLCall(glBindBuffer(GL_ARRAY_BUFFER, 0));
}


void OpenGLWrapper::BindShaderProgram() const
{
	GLCall(glUseProgram(m_ShaderProgramID));
}

void OpenGLWrapper::UnBindShaderProgram() const
{
	GLCall(glUseProgram(0));
}


ShaderSource OpenGLWrapper::ParseShader(const std::string& filePath)
{
	std::ifstream stream(filePath);

	enum class ShaderType {

		NONE = -1, VERTEX = 0, FRAGMENT = 1, COMPUTE = 2

	};

	std::string line;
	std::stringstream ss[3];
	ShaderType type = ShaderType::NONE;

	while (getline(stream, line))
	{
		if (line.find("#shader") != std::string::npos)
		{
			if (line.find("compute") != std::string::npos)
			{
				type = ShaderType::COMPUTE;
			}
			else if (line.find("vertex") != std::string::npos)
			{
				type = ShaderType::VERTEX;
			}
			else if (line.find("fragment") != std::string::npos)
			{
				type = ShaderType::FRAGMENT;
			}
		}
		else
		{
			ss[(int)type] << line << "\n";
		}
	}
	return { ss[0].str(), ss[1].str(), ss[2].str() };
}

unsigned int OpenGLWrapper::CreateShader(const std::string& vs, const std::string& fs)
{

	GLCall(unsigned int program = glCreateProgram());

	unsigned int vertex = CompileShader(vs, GL_VERTEX_SHADER);
	unsigned int fragment = CompileShader(fs, GL_FRAGMENT_SHADER);
	GLCall(glAttachShader(program, vertex));
	GLCall(glAttachShader(program, fragment));
	GLCall(glLinkProgram(program));
	GLCall(glValidateProgram(program));

	GLCall(glDeleteShader(vertex));
	GLCall(glDeleteShader(fragment));

	return program;
}

unsigned int OpenGLWrapper::CompileShader(const std::string& source, unsigned int type)
{

	std::string shaderType;
	switch (type)
	{
	case GL_VERTEX_SHADER:
		shaderType = "Vertex";
		break;
	case GL_FRAGMENT_SHADER:
		shaderType = "Fragment";
		break;
	case GL_COMPUTE_SHADER:
		shaderType = "Compute";
		break;
	default:
		break;
	}
	unsigned int shader;
	GLCall(shader = glCreateShader(type));
	const char* sc = source.c_str();
	GLCall(glShaderSource(shader, 1, &sc, nullptr));
	GLCall(glCompileShader(shader));

	int result;
	GLCall(glGetShaderiv(shader, GL_COMPILE_STATUS, &result));
	if (result == GL_FALSE)
	{
		int length;
		GLCall(glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length));
		char* message = (char*)alloca(length * sizeof(char));
		GLCall(glGetShaderInfoLog(shader, length, &length, message));
		std::cout << "Failed to compile " << shaderType << " shader" << std::endl;
		std::cout << message << std::endl;
		GLCall(glDeleteShader(shader));
		return 0;

	}
	return shader;
}

void OpenGLWrapper::SetUniform1i(int location)
{
	GLCall(glUniform1i(location, 0));
}

int OpenGLWrapper::GetUniformLocation(const std::string& name)
{
	if (m_UniformLocationCache.find(name) != m_UniformLocationCache.end())
		return m_UniformLocationCache[name];
	GLCall(int location = glGetUniformLocation(m_ShaderProgramID, name.c_str()));
	if (location == -1)
	{
		std::cout << "Warning: uniform '" << name << "' doesn't exist!" << std::endl;
	}
	m_UniformLocationCache[name] = location;
	return location;
}

unsigned int OpenGLWrapper::GetRenderID() const {
	return m_ShaderProgramID;
}

unsigned int OpenGLWrapper::GetTextureID() const
{
	return m_TextureID;
}

void OpenGLWrapper::DeleteTexture() {
	if (m_TextureID > 0) GLCall(glDeleteTextures(1, &m_TextureID));
}

void OpenGLWrapper::printInfo()
{
	int majorVersion, minorVersion;
	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
}


OpenGLWrapper::~OpenGLWrapper()
{
	GLCall(glDeleteProgram(m_ShaderProgramID));
}

void OpenGLWrapper::Draw() {
	GLCall(glDrawArrays(GL_TRIANGLE_FAN, 0, 4));
}

void OpenGLWrapper::DefineTexture(int width, int height, const void* data) {

	GLCall(glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, width,height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, data));
}

void OpenGLWrapper::BindTexture() const {

	GLCall(glActiveTexture(GL_TEXTURE0));
	GLCall(glBindTexture(GL_TEXTURE_2D,m_TextureID));
}