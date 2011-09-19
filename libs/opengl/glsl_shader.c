#include <opengl/glsl_shader.h>

#include <common/error.h>

#include <stdlib.h>

char* file_to_string(const char* path) {
	FILE* fd;
	long len, r;
	char* str;
 
	if (!(fd = fopen(path, "r"))) {
		exit_with_error("Can't open file '%s' for reading\n", path);
	}
 
	fseek(fd, 0, SEEK_END);
	len = ftell(fd);
 
	fseek(fd, 0, SEEK_SET);
	if (!(str = (char*) malloc(len * sizeof(char)))) {
        exit_with_error("Can't malloc space for '%s'\n", path);
	}
 
	r = fread(str, sizeof(char), len, fd);
	str[r - 1] = '\0'; /* Shader sources have to term with null */
	fclose(fd);
	return str;
}

void print_shader_info_log(GLuint shader) {
    GLint infologLength = 0, charsWritten = 0;
    GLchar *infoLog;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infologLength);
    if (infologLength > 0) {
        infoLog = (GLchar *)malloc(infologLength);
        if (infoLog==NULL) {
            exit_with_error("Could not allocate InfoLog buffer for shader: %u", shader);
        }
        glGetShaderInfoLog(shader, infologLength, &charsWritten, infoLog);
        if (charsWritten > 0)
            fprintf(stderr, "Shader InfoLog: %u\n%s", shader, infoLog);
        free(infoLog);
    }
}

void print_program_info_log(GLuint program) {
    GLint infologLength = 0, charsWritten = 0;
    GLchar* infoLog;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infologLength);
    if (infologLength > 0) {
        infoLog = (GLchar *)malloc(infologLength);
        if (infoLog==NULL) {
            exit_with_error("Could not allocate InfoLog buffer for program: %u", program);
        }
        glGetProgramInfoLog(program, infologLength, &charsWritten, infoLog);
        if (charsWritten > 0)
            fprintf(stderr, "Program InfoLog: %u\n%s", program, infoLog);
        free(infoLog);
    }
}

GLuint load_shader(const char* file, int type) {
    GLuint shader = glCreateShader(type);
    const char* source = file_to_string(file);
    glShaderSource(shader, 1, &source, NULL);

    glCompileShader(shader);
    GLint compiled;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    print_shader_info_log(shader);
    if (compiled==0) {
        exit_with_error("Failed compiling shader program: %s\n", file);
    }
    return shader;
}

void glsl_shader_init(glsl_shader* shader, 
                      const char* vertex_program_file,
                      const char* fragment_program_file) {
    shader->shaderProgram = 0;
    shader->vertexShaderId = 0;
    shader->fragmentShaderId = 0;


    shader->shaderProgram = glCreateProgram();

    shader->vertexShaderId = load_shader(vertex_program_file, GL_VERTEX_SHADER);
    if (shader->vertexShaderId == 0) {
        exit_with_error("Failed loading vertex shader: %s", fragment_program_file);
    }
    glAttachShader(shader->shaderProgram, shader->vertexShaderId);

    shader->fragmentShaderId = load_shader(fragment_program_file, GL_FRAGMENT_SHADER);
    if (shader->fragmentShaderId == 0) {
        exit_with_error("Failed loading fragment shader: %s", fragment_program_file);
    }
    glAttachShader(shader->shaderProgram, shader->fragmentShaderId);
            
    glLinkProgram(shader->shaderProgram);
    GLint linked;
    glGetProgramiv(shader->shaderProgram, GL_LINK_STATUS, &linked);
            
    if (linked == 0) {
        exit_with_error("Could not link shader program");
    }

    CHECK_FOR_GL_ERROR();
    print_program_info_log(shader->shaderProgram);
}

void glsl_shader_apply(glsl_shader* shader) {
    glUseProgram(shader->shaderProgram);
}

void glsl_shader_release(glsl_shader* shader) {
    glUseProgram(0);
}
