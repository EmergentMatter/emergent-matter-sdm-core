"""Execute emitted shaders using a headless software OpenGL ES context.

Uses the system EGL/GLES libraries through ctypes so parity tests need no new
Python dependency. Context availability may skip locally, but is required in CI.
Shader compilation and execution errors always fail the test.
"""

from __future__ import annotations

import ctypes as ct
import ctypes.util
import os

import numpy as np


class ShaderRuntime:
    def __init__(self):
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "true"
        self.egl = ct.CDLL(ctypes.util.find_library("EGL") or "libEGL.so.1")
        self.gl = ct.CDLL(ctypes.util.find_library("GLESv2") or "libGLESv2.so.2")
        ptr, integer, uint = ct.c_void_p, ct.c_int, ct.c_uint
        self._bind(self.egl, "eglGetPlatformDisplay", ptr, uint, ptr, ct.POINTER(integer))
        self._bind(self.egl, "eglInitialize", uint, ptr, ptr, ptr)
        self._bind(self.egl, "eglBindAPI", uint, uint)
        self._bind(self.egl, "eglChooseConfig", uint, ptr, ptr, ptr, integer, ptr)
        self._bind(self.egl, "eglCreateContext", ptr, ptr, ptr, ptr, ptr)
        self._bind(self.egl, "eglMakeCurrent", uint, ptr, ptr, ptr, ptr)
        self._bind(self.egl, "eglDestroyContext", uint, ptr, ptr)
        self._bind(self.egl, "eglTerminate", uint, ptr)
        self.display = self.egl.eglGetPlatformDisplay(0x31DD, None, None)
        if not self.egl.eglInitialize(self.display, None, None):
            raise RuntimeError("Software EGL display unavailable")
        self.context = None
        try:
            self.egl.eglBindAPI(0x30A0)
            config, count = ptr(), integer()
            attrs = (integer * 7)(0x3033, 1, 0x3040, 0x40, 0x3024, 8, 0x3038)
            if (
                not self.egl.eglChooseConfig(
                    self.display, attrs, ct.byref(config), 1, ct.byref(count)
                )
                or not count.value
            ):
                raise RuntimeError("Software EGL ES3 config unavailable")
            ctx_attrs = (integer * 3)(0x3098, 3, 0x3038)
            self.context = self.egl.eglCreateContext(self.display, config, None, ctx_attrs)
            if not self.context or not self.egl.eglMakeCurrent(
                self.display, None, None, self.context
            ):
                raise RuntimeError("Software EGL ES3 context unavailable")
        except Exception:
            self.close()
            raise
        signatures = {
            "glCreateShader": (uint, uint),
            "glShaderSource": (None, uint, integer, ptr, ptr),
            "glCompileShader": (None, uint),
            "glGetShaderiv": (None, uint, uint, ptr),
            "glGetShaderInfoLog": (None, uint, integer, ptr, ptr),
            "glCreateProgram": (uint,),
            "glAttachShader": (None, uint, uint),
            "glLinkProgram": (None, uint),
            "glGetProgramiv": (None, uint, uint, ptr),
            "glGetProgramInfoLog": (None, uint, integer, ptr, ptr),
            "glUseProgram": (None, uint),
            "glGetUniformLocation": (integer, uint, ct.c_char_p),
            "glUniform1f": (None, integer, ct.c_float),
            "glUniform1i": (None, integer, integer),
            "glGenTextures": (None, integer, ptr),
            "glActiveTexture": (None, uint),
            "glBindTexture": (None, uint, uint),
            "glTexParameteri": (None, uint, uint, integer),
            "glTexImage2D": (
                None,
                uint,
                integer,
                integer,
                integer,
                integer,
                integer,
                uint,
                uint,
                ptr,
            ),
            "glDeleteTextures": (None, integer, ptr),
            "glGenBuffers": (None, integer, ptr),
            "glBindBuffer": (None, uint, uint),
            "glBufferData": (None, uint, ct.c_ssize_t, ptr, uint),
            "glBindBufferBase": (None, uint, uint, uint),
            "glDispatchCompute": (None, uint, uint, uint),
            "glMemoryBarrier": (None, uint),
            "glMapBufferRange": (ptr, uint, ct.c_ssize_t, ct.c_ssize_t, uint),
            "glUnmapBuffer": (uint, uint),
            "glDeleteBuffers": (None, integer, ptr),
            "glDeleteProgram": (None, uint),
            "glDeleteShader": (None, uint),
        }
        for name, (result, *args) in signatures.items():
            self._bind(self.gl, name, result, *args)

    @staticmethod
    def _bind(lib, name, result, *args):
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = result, args

    def close(self):
        self.egl.eglMakeCurrent(self.display, None, None, None)
        if self.context:
            self.egl.eglDestroyContext(self.display, self.context)
        self.egl.eglTerminate(self.display)

    def evaluate(self, emission, points, uniforms=None, expression=None):
        """Return vec4(rest point, scene distance), or a supplied vec4 expression."""
        gl = self.gl
        expression = expression or "vec4(sdm_rest_point(p, 0), sdf_scene(p))"
        source = (
            "#version 310 es\nprecision highp float;\nprecision highp int;\n"
            "precision highp sampler2D;\n"
            + emission.lib_source
            + emission.scene_source
            + "\nlayout(local_size_x=1) in;\n"
            "layout(std430, binding=0) readonly buffer Input { vec4 points[]; };\n"
            "layout(std430, binding=1) writeonly buffer Output { vec4 values[]; };\n"
            "void main() { uint i=gl_GlobalInvocationID.x; vec3 p=points[i].xyz; "
            f"values[i]={expression}; }}"
        ).encode()
        shader = gl.glCreateShader(0x91B9)
        program = gl.glCreateProgram()
        buffers = (ct.c_uint * 2)()
        texture = ct.c_uint()
        try:
            text = ct.c_char_p(source)
            gl.glShaderSource(shader, 1, ct.byref(text), None)
            gl.glCompileShader(shader)
            status = ct.c_int()
            gl.glGetShaderiv(shader, 0x8B81, ct.byref(status))
            if not status.value:
                log = ct.create_string_buffer(65536)
                gl.glGetShaderInfoLog(shader, len(log), None, log)
                raise AssertionError(log.value.decode())
            gl.glAttachShader(program, shader)
            gl.glLinkProgram(program)
            gl.glGetProgramiv(program, 0x8B82, ct.byref(status))
            if not status.value:
                log = ct.create_string_buffer(65536)
                gl.glGetProgramInfoLog(program, len(log), None, log)
                raise AssertionError(log.value.decode())
            gl.glUseProgram(program)
            values = {u.name: u.initial for u in emission.uniforms}
            values.update(uniforms or {})
            for name, value in values.items():
                gl.glUniform1f(gl.glGetUniformLocation(program, name.encode()), value)
            sweep_table = getattr(emission, "sweep_table", ())
            if sweep_table:
                width = emission.sweep_tex_width
                texels = np.asarray(sweep_table, dtype=np.float32).reshape(-1, 4)
                height = (len(texels) + width - 1) // width
                pixels = np.zeros((height * width, 4), dtype=np.float32)
                pixels[: len(texels)] = texels
                gl.glGenTextures(1, ct.byref(texture))
                # A nonzero unit catches a sampler left at its default binding.
                gl.glActiveTexture(0x84C0 + 3)
                gl.glBindTexture(0x0DE1, texture)
                gl.glTexParameteri(0x0DE1, 0x2801, 0x2600)
                gl.glTexParameteri(0x0DE1, 0x2800, 0x2600)
                gl.glTexImage2D(
                    0x0DE1, 0, 0x8814, width, height, 0, 0x1908, 0x1406, pixels.ctypes.data
                )
                gl.glUniform1i(gl.glGetUniformLocation(program, b"u_sdm_sweep"), 3)
            data = np.zeros((len(points), 4), dtype=np.float32)
            data[:, :3] = points
            gl.glGenBuffers(2, buffers)
            for i in range(2):
                gl.glBindBuffer(0x90D2, buffers[i])
                gl.glBufferData(0x90D2, data.nbytes, data.ctypes.data if i == 0 else None, 0x88E8)
                gl.glBindBufferBase(0x90D2, i, buffers[i])
            gl.glDispatchCompute(len(points), 1, 1)
            # Make shader writes visible to the following CPU buffer mapping.
            gl.glMemoryBarrier(0x2200)
            address = gl.glMapBufferRange(0x90D2, 0, data.nbytes, 1)
            if not address:
                raise AssertionError("Could not read shader output")
            result = np.ctypeslib.as_array((ct.c_float * data.size).from_address(address)).copy()
            gl.glUnmapBuffer(0x90D2)
            return result.reshape(data.shape)
        finally:
            gl.glDeleteTextures(1, ct.byref(texture))
            gl.glDeleteBuffers(2, buffers)
            gl.glDeleteProgram(program)
            gl.glDeleteShader(shader)
