#include "Application.h"

Application::Application()
{
	image = new Image("src/Images/lena.bmp");
	if (!glfwInit())
		std::cout << "GLFW couldn't initialize properly" << std::endl;

	window = glfwCreateWindow(image->GetWidth(), image->GetHeight(), "Denoise", NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		std::cout << "No window available" << std::endl;
	}

	glfwMakeContextCurrent(window);

	glfwSwapInterval(1);
	if (glewInit() != GLEW_OK)
		std::cout << "Baj van";

	openglwrapper = new OpenGLWrapper("Shader.shader");
	openglwrapper->BindShaderProgram();
	openglwrapper->printInfo();
	menu= new ImGuiMenu(window, openGlRenderer);
	openGlRenderer = new OpenGLRenderer(image->GetWidth(), image->GetHeight(), openglwrapper, nullptr, image);
	menu->SetOpenGLRenderer(openGlRenderer);
	menu->Draw();
	openGlRenderer->SetMenu(menu);
	openGlRenderer->UpdateTexture();

}

void Application::Run()
{
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
		if (GLFW_PRESS == glfwGetKey(window, GLFW_KEY_ESCAPE))
		{
			glfwSetWindowShouldClose(window, 1);
		}
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();


		openGlRenderer->Draw();

		menu->Update();

		glfwSwapBuffers(window);

	}
}

Application::~Application()
{
	menu->Kill();
	glfwDestroyWindow(window);
	glfwTerminate();
	delete window;
	delete image;
	delete openglwrapper;
	delete openGlRenderer;
	delete menu;
}