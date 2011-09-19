#include <opengl/image2d_opengl.h>

#include <opengl/opengl_utilities.h>
#include <common/error.h>

unsigned int image2d_upload_to_gpu(image2d* i) {
    unsigned int id;
	glGenTextures(1, &id);
	glBindTexture(GL_TEXTURE_2D, id);

    unsigned int channels = i->channels;
    GLenum type;
    if (channels == 1)
        type = GL_LUMINANCE;
    else if(channels == 3)
        type = GL_RGB;
    else if(channels == 4)
        type = GL_RGBA;
    else {
        exit_with_error("cannot handle channels= %i\n", channels);
    }

	glTexImage2D(GL_TEXTURE_2D, 0,
                 type, i->width, i->height,
                 0, type, GL_UNSIGNED_BYTE, i->data);
	glBindTexture(GL_TEXTURE_2D, id);

	/* Linear Filtering */
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	/* glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); */
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

    /* glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP,    GL_TRUE); */
    glGenerateMipmap(GL_TEXTURE_2D);

    /* Clamp to edge */
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return id;
}

void image2d_remove_from_gpu_memory(unsigned int textureid) {
    glDeleteTextures(1, &textureid);
}
