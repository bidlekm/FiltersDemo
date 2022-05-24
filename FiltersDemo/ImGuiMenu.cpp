#include "ImGuiMenu.h"

int ImGuiMenu::ID;


void ImGuiMenu::Draw()
{
	IMGUI_CHECKVERSION();
	ImGuiContext* context = ImGui::CreateContext();
	ImGui::SetCurrentContext(context);

	// Setup Dear ImGui style
	ImGui::StyleColorsDark();
	const char* glsl_version = "#version 330";
	// Setup Platform/Renderer backends
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init(glsl_version);
	Theme();
}

ImGuiMenu::ImGuiMenu(GLFWwindow* _window, OpenGLRenderer* _openGlRenderer): window(_window), openGlRenderer(_openGlRenderer)
{
	currentNoiseType = noiseTypes[0];
	currentDenoiseAlgorithm = denoiseAlgorithms[0];
}

float ImGuiMenu::AddFloat(float value, float min, float max)
{
	ImGui::PushID(ImGuiMenu::ID++);
	float temp = value;
	ImGui::SliderFloat("", &temp, min, max);
	ImGui::PopID();
	return temp;
}

glm::vec3 ImGuiMenu::AddVec3(glm::vec3 value, float min, float max)
{
	ImGui::PushID(ImGuiMenu::ID++);
	glm::vec3 temp = value;
	ImGui::SliderFloat3("", &temp.x, min, max);
	ImGui::PopID();
	return temp;
}

glm::vec2 ImGuiMenu::AddVec2(glm::vec2 value, float min, float max)
{
	ImGui::PushID(ImGuiMenu::ID++);
	glm::vec2 temp = value;
	ImGui::SliderFloat2("", &temp.x, min, max);
	ImGui::PopID();
	return temp;
}

glm::vec3 ImGuiMenu::AddColor3f(glm::vec3 value)
{
	ImGui::PushID(ImGuiMenu::ID++);
	glm::vec3 temp = value;
	ImGui::ColorPicker3("Color", &temp.x, ImGuiColorEditFlags_NoOptions | ImGuiColorEditFlags_NoInputs);
	ImGui::PopID();
	return temp;
}

void ImGuiMenu::SetOpenGLRenderer(OpenGLRenderer* _openglrenderer)
{
	openGlRenderer = _openglrenderer;
}

void ImGuiMenu::Update()
{
	{
		ImGui::Begin("Menu");

		if (ImGui::BeginCombo("Noise types", currentNoiseType)) // The second parameter is the label previewed before opening the combo.
		{
			for (int n = 0; n < IM_ARRAYSIZE(noiseTypes); n++)
			{
				bool is_selected = (currentNoiseType == noiseTypes[n]); // You can store your selection however you want, outside or inside your objects
				if (ImGui::Selectable(noiseTypes[n], is_selected))
					currentNoiseType = noiseTypes[n];
					if (is_selected)
						ImGui::SetItemDefaultFocus();   // You may set the initial focus when opening the combo (scrolling + for keyboard navigation support)
			}
			ImGui::EndCombo();
		}
		if (ImGui::Button("Noise"))
		{
			Image* image = openGlRenderer->GetImage();
			AddNoiseToImage(image->GetWidth(), image->GetHeight(), image->GetData());
			openGlRenderer->UpdateTexture();
		}

		ImGui::Separator;

		if (ImGui::BeginCombo("Denoising algorithms", currentDenoiseAlgorithm)) 
		{
			for (int n = 0; n < IM_ARRAYSIZE(denoiseAlgorithms); n++)
			{
				bool is_selected = (currentDenoiseAlgorithm == denoiseAlgorithms[n]);
				if (ImGui::Selectable(denoiseAlgorithms[n], is_selected))
					currentDenoiseAlgorithm = denoiseAlgorithms[n];
				if (is_selected)
					ImGui::SetItemDefaultFocus();   
			}
			ImGui::EndCombo();
		}
		if (ImGui::Button("Denoise"))
		{
			Image* image = openGlRenderer->GetImage();
			PrimalDualDenoise(image->GetWidth(), image->GetHeight(), image->GetData());
			//GradientDescentDenoise(image->GetWidth(), image->GetHeight(), image->GetData());
			openGlRenderer->UpdateTexture();
		}

		ImGui::End();
	}
	ImGui::Update();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void ImGuiMenu::Theme()
{

	ImGuiStyle* style = &ImGui::GetStyle();
	style->WindowBorderSize = 0;
	style->WindowTitleAlign = ImVec2(0.5, 0.5);
	style->WindowMinSize = ImVec2(300, 300);
	style->FramePadding = ImVec2(10, 10);
	style->Colors[ImGuiCol_TitleBg] = ImColor(255, 101, 53, 255);
	style->Colors[ImGuiCol_TitleBgActive] = ImColor(255, 101, 53, 255);
	style->Colors[ImGuiCol_TitleBgCollapsed] = ImColor(0, 0, 0, 130);

	style->Colors[ImGuiCol_Button] = ImColor(31, 30, 31, 255);
	style->Colors[ImGuiCol_ButtonActive] = ImColor(31, 30, 31, 255);
	style->Colors[ImGuiCol_ButtonHovered] = ImColor(41, 40, 41, 255);

	style->Colors[ImGuiCol_FrameBg] = ImColor(37, 36, 37, 255);
	style->Colors[ImGuiCol_FrameBgActive] = ImColor(37, 36, 37, 255);
	style->Colors[ImGuiCol_FrameBgHovered] = ImColor(37, 36, 37, 255);

	style->Colors[ImGuiCol_Header] = ImColor(0, 0, 0, 0);
	style->Colors[ImGuiCol_HeaderActive] = ImColor(0, 0, 0, 0);
	style->Colors[ImGuiCol_HeaderHovered] = ImColor(46, 46, 46, 255);

}

void ImGuiMenu::Kill()
{
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}


ImGuiMenu::~ImGuiMenu()
{

}

