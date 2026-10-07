mod implicit;

use std::fmt::Write;

pub use implicit::*;

// define constants
/// No alpha shader define
pub const NO_ALPHA: &str = "NO_ALPHA";
/// External texture shader define
pub const EXTERNAL: &str = "EXTERNAL";
/// Debug flags shader define
pub const DEBUG_FLAGS: &str = "DEBUG_FLAGS";

use super::*;

/// Compiles a shader variant.
///
/// # Safety
///
/// You must call this only when it is safe to compile shaders with GL.
pub unsafe fn compile_shader(
    gl: &ffi::Gles2,
    variant: ffi::types::GLuint,
    src: &str,
) -> Result<ffi::types::GLuint, GlesError> {
    let shader = gl.CreateShader(variant);
    if shader == 0 {
        return Err(GlesError::CreateShaderObject);
    }

    gl.ShaderSource(
        shader,
        1,
        &src.as_ptr() as *const *const u8 as *const *const ffi::types::GLchar,
        &(src.len() as i32) as *const _,
    );
    gl.CompileShader(shader);

    let mut status = ffi::FALSE as i32;
    gl.GetShaderiv(shader, ffi::COMPILE_STATUS, &mut status as *mut _);
    if status == ffi::FALSE as i32 {
        let mut max_len = 0;
        gl.GetShaderiv(shader, ffi::INFO_LOG_LENGTH, &mut max_len as *mut _);

        let mut error = Vec::with_capacity(max_len as usize);
        let mut len = 0;
        gl.GetShaderInfoLog(
            shader,
            max_len as _,
            &mut len as *mut _,
            error.as_mut_ptr() as *mut _,
        );
        error.set_len(len as usize);

        error!(
            "[GL] {}",
            std::str::from_utf8(&error).unwrap_or("<Error Message no utf8>")
        );

        gl.DeleteShader(shader);
        return Err(GlesError::ShaderCompileError);
    }

    Ok(shader)
}

/// The name of the color output a GLSL ES 3.00 fragment shader declares
/// (`out vec4 name;`, with or without a layout qualifier), or `None` for a
/// shader that writes `gl_FragColor`.
fn fragment_output(frag_src: &str) -> Option<&str> {
    if !frag_src.trim_start().starts_with("#version 300 es") {
        return None;
    }
    frag_src.lines().find_map(|line| {
        let line = line.split("//").next()?.trim();
        let line = match line.strip_prefix("layout") {
            Some(rest) => rest.split_once(')')?.1.trim_start(),
            None => line,
        };
        let declaration = line.strip_prefix("out")?.strip_suffix(';')?;
        let mut tokens = declaration.split_whitespace();
        let name = tokens.next_back()?;
        tokens.any(|token| token == "vec4").then_some(name)
    })
}

/// The fragment source that is compiled for `frag_src`: its `main` renamed and
/// `color.frag`'s managed-color `main` appended. [`link_program`] and
/// [`link_program_deferred`] share it, so both name the same entry in the
/// driver's shader cache.
fn managed_fragment(frag_src: &str) -> String {
    let mut managed_source = frag_src.replace("void main(", "void smithay_original_main(");
    match fragment_output(frag_src) {
        // GLSL ES 3.00 has no gl_FragColor: convert the shader's own output.
        Some(output) => managed_source.push_str(&include_str!("color.frag").replace("gl_FragColor", output)),
        None => managed_source.push_str(include_str!("color.frag")),
    }
    managed_source
}

/// Compiles and links a shader program.
///
/// # Safety
///
/// You must call this only when it is safe to compile and link shaders with GL.
pub unsafe fn link_program(
    gl: &ffi::Gles2,
    vert_src: &str,
    frag_src: &str,
) -> Result<ffi::types::GLuint, GlesError> {
    let vert = compile_shader(gl, ffi::VERTEX_SHADER, vert_src)?;
    let frag = compile_shader(gl, ffi::FRAGMENT_SHADER, &managed_fragment(frag_src))?;
    let program = gl.CreateProgram();
    gl.AttachShader(program, vert);
    gl.AttachShader(program, frag);
    gl.LinkProgram(program);
    gl.DetachShader(program, vert);
    gl.DetachShader(program, frag);
    gl.DeleteShader(vert);
    gl.DeleteShader(frag);

    let mut status = ffi::FALSE as i32;
    gl.GetProgramiv(program, ffi::LINK_STATUS, &mut status as *mut _);
    if status == ffi::FALSE as i32 {
        let mut max_len = 0;
        gl.GetProgramiv(program, ffi::INFO_LOG_LENGTH, &mut max_len as *mut _);

        let mut error = Vec::with_capacity(max_len as usize);
        let mut len = 0;
        gl.GetProgramInfoLog(
            program,
            max_len as _,
            &mut len as *mut _,
            error.as_mut_ptr() as *mut _,
        );
        error.set_len(len as usize);

        error!(
            "[GL] {}",
            std::str::from_utf8(&error).unwrap_or("<Error Message no utf8>")
        );

        gl.DeleteProgram(program);
        return Err(GlesError::ProgramLinkError);
    }

    Ok(program)
}

/// Whether the space-separated GL extension `list` names `extension`.
fn names_extension(list: &str, extension: &str) -> bool {
    list.split(' ').any(|name| name == extension)
}

/// A shader program that is still compiling or linking, started by
/// [`link_program_deferred`].
#[derive(Debug)]
pub struct DeferredProgram {
    program: ffi::types::GLuint,
    shaders: [ffi::types::GLuint; 2],
    /// `GL_KHR_parallel_shader_compile`: completion can be asked without waiting.
    parallel: bool,
}

/// What [`DeferredProgram::poll`] found.
#[derive(Debug)]
pub enum DeferredLink {
    /// Still compiling or linking: poll again later.
    Pending(DeferredProgram),
    /// The linked program, as [`link_program`] returns it.
    Linked(ffi::types::GLuint),
    /// Compiling or linking failed; the GL objects are deleted.
    Failed(GlesError),
}

/// Starts compiling and linking a shader program without waiting for it, from
/// the same sources [`link_program`] compiles.
///
/// With `GL_KHR_parallel_shader_compile` the context's compiler thread count
/// is raised to the driver's maximum and [`DeferredProgram::poll`] never
/// waits. How much of the work leaves the calling thread is the driver's
/// choice: Mesa still parses and links GLSL inside `glLinkProgram` and defers
/// only its code generation. Without the extension the driver may compile
/// inside this call or inside the first poll, which then blocks like
/// [`link_program`].
///
/// # Safety
///
/// You must call this only when it is safe to compile and link shaders with GL.
pub unsafe fn link_program_deferred(
    gl: &ffi::Gles2,
    vert_src: &str,
    frag_src: &str,
) -> Result<DeferredProgram, GlesError> {
    let extensions = gl.GetString(ffi::EXTENSIONS) as *const c_char;
    let parallel = !extensions.is_null()
        && names_extension(
            &CStr::from_ptr(extensions).to_string_lossy(),
            "GL_KHR_parallel_shader_compile",
        );
    if parallel {
        // As many compiler threads as the driver has.
        gl.MaxShaderCompilerThreadsKHR(ffi::types::GLuint::MAX);
    }

    let frag_src = managed_fragment(frag_src);
    let mut shaders = [0; 2];
    for (shader, (variant, src)) in shaders.iter_mut().zip([
        (ffi::VERTEX_SHADER, vert_src),
        (ffi::FRAGMENT_SHADER, frag_src.as_str()),
    ]) {
        *shader = gl.CreateShader(variant);
        if *shader == 0 {
            gl.DeleteShader(shaders[0]);
            return Err(GlesError::CreateShaderObject);
        }
        gl.ShaderSource(
            *shader,
            1,
            &src.as_ptr() as *const *const u8 as *const *const ffi::types::GLchar,
            &(src.len() as i32) as *const _,
        );
        // No status query here: it would wait for the compiler.
        gl.CompileShader(*shader);
    }
    let program = gl.CreateProgram();
    gl.AttachShader(program, shaders[0]);
    gl.AttachShader(program, shaders[1]);
    gl.LinkProgram(program);

    Ok(DeferredProgram {
        program,
        shaders,
        parallel,
    })
}

impl DeferredProgram {
    /// Asks whether the program has linked. Never waits for the compiler when
    /// the context has `GL_KHR_parallel_shader_compile`.
    ///
    /// # Safety
    ///
    /// The context of [`link_program_deferred`] must be current.
    pub unsafe fn poll(self, gl: &ffi::Gles2) -> DeferredLink {
        if self.parallel {
            // Any other query of the program or its shaders waits for the link.
            let mut complete = ffi::FALSE as i32;
            gl.GetProgramiv(self.program, ffi::COMPLETION_STATUS_KHR, &mut complete as *mut _);
            if complete == ffi::FALSE as i32 {
                return DeferredLink::Pending(self);
            }
        }
        match self.wait(gl) {
            Ok(program) => DeferredLink::Linked(program),
            Err(err) => DeferredLink::Failed(err),
        }
    }

    /// Waits for the link, as [`link_program`] does.
    ///
    /// # Safety
    ///
    /// The context of [`link_program_deferred`] must be current.
    pub unsafe fn wait(self, gl: &ffi::Gles2) -> Result<ffi::types::GLuint, GlesError> {
        let Self { program, shaders, .. } = self;
        let mut status = ffi::FALSE as i32;
        gl.GetProgramiv(program, ffi::LINK_STATUS, &mut status as *mut _);
        let result = if status == ffi::FALSE as i32 {
            // A shader that failed to compile fails the link; report the cause.
            let mut err = GlesError::ProgramLinkError;
            for shader in shaders {
                let mut compiled = ffi::FALSE as i32;
                gl.GetShaderiv(shader, ffi::COMPILE_STATUS, &mut compiled as *mut _);
                if compiled == ffi::FALSE as i32 {
                    error!("[GL] {}", info_log(gl, shader, false));
                    err = GlesError::ShaderCompileError;
                }
            }
            if matches!(err, GlesError::ProgramLinkError) {
                error!("[GL] {}", info_log(gl, program, true));
            }
            Err(err)
        } else {
            Ok(program)
        };
        for shader in shaders {
            gl.DetachShader(program, shader);
            gl.DeleteShader(shader);
        }
        if result.is_err() {
            gl.DeleteProgram(program);
        }
        result
    }

    /// Deletes the program without waiting for it.
    ///
    /// # Safety
    ///
    /// The context of [`link_program_deferred`] must be current.
    pub unsafe fn discard(self, gl: &ffi::Gles2) {
        for shader in self.shaders {
            gl.DeleteShader(shader);
        }
        gl.DeleteProgram(self.program);
    }
}

/// The info log of a program or shader.
unsafe fn info_log(gl: &ffi::Gles2, object: ffi::types::GLuint, program: bool) -> String {
    let mut max_len = 0;
    if program {
        gl.GetProgramiv(object, ffi::INFO_LOG_LENGTH, &mut max_len as *mut _);
    } else {
        gl.GetShaderiv(object, ffi::INFO_LOG_LENGTH, &mut max_len as *mut _);
    }
    let mut log = Vec::<u8>::with_capacity(max_len as usize);
    let mut len = 0;
    if program {
        gl.GetProgramInfoLog(
            object,
            max_len as _,
            &mut len as *mut _,
            log.as_mut_ptr() as *mut _,
        );
    } else {
        gl.GetShaderInfoLog(
            object,
            max_len as _,
            &mut len as *mut _,
            log.as_mut_ptr() as *mut _,
        );
    }
    log.set_len(len as usize);
    String::from_utf8(log).unwrap_or_else(|_| "<Error Message no utf8>".into())
}

pub(super) unsafe fn texture_program(
    gl: &ffi::Gles2,
    src: &str,
    additional_uniforms: &[UniformName<'_>],
    destruction_callback_sender: Sender<CleanupResource>,
) -> Result<GlesTexProgram, GlesError> {
    let create_variant = |defines: &[&str]| -> Result<GlesTexProgramVariant, GlesError> {
        let shader = src.replace(
            "//_DEFINES_",
            &defines.iter().fold(String::new(), |mut shader, define| {
                let _ = writeln!(&mut shader, "#define {define}");
                shader
            }),
        );
        let debug_shader = src.replace(
            "//_DEFINES_",
            &defines
                .iter()
                .chain(&[shaders::DEBUG_FLAGS])
                .fold(String::new(), |mut shader, define| {
                    let _ = writeln!(shader, "#define {define}");
                    shader
                }),
        );

        let program = unsafe { link_program(gl, shaders::VERTEX_SHADER, &shader)? };
        let debug_program = unsafe { link_program(gl, shaders::VERTEX_SHADER, debug_shader.as_ref())? };

        let vert = c"vert";
        let vert_position = c"vert_position";
        let tex = c"tex";
        let matrix = c"matrix";
        let tex_matrix = c"tex_matrix";
        let alpha = c"alpha";
        let tint = c"tint";

        Ok(GlesTexProgramVariant {
            normal: GlesTexProgramInternal {
                program,
                uniform_color_transform: gl.GetUniformLocation(program, c"smithay_color[0]".as_ptr()),
                uniform_tex: gl.GetUniformLocation(program, tex.as_ptr() as *const ffi::types::GLchar),
                uniform_matrix: gl.GetUniformLocation(program, matrix.as_ptr() as *const ffi::types::GLchar),
                uniform_tex_matrix: gl
                    .GetUniformLocation(program, tex_matrix.as_ptr() as *const ffi::types::GLchar),
                uniform_alpha: gl.GetUniformLocation(program, alpha.as_ptr() as *const ffi::types::GLchar),
                attrib_vert: gl.GetAttribLocation(program, vert.as_ptr() as *const ffi::types::GLchar),
                attrib_vert_position: gl
                    .GetAttribLocation(program, vert_position.as_ptr() as *const ffi::types::GLchar),
                additional_uniforms: additional_uniforms
                    .iter()
                    .map(|uniform| {
                        let name = CString::new(uniform.name.as_bytes()).expect("Interior null in name");
                        let location =
                            gl.GetUniformLocation(program, name.as_ptr() as *const ffi::types::GLchar);
                        (
                            uniform.name.clone().into_owned(),
                            UniformDesc {
                                location,
                                type_: uniform.type_,
                            },
                        )
                    })
                    .collect(),
            },
            debug: GlesTexProgramInternal {
                program: debug_program,
                uniform_color_transform: gl.GetUniformLocation(debug_program, c"smithay_color[0]".as_ptr()),
                uniform_tex: gl.GetUniformLocation(debug_program, tex.as_ptr() as *const ffi::types::GLchar),
                uniform_matrix: gl
                    .GetUniformLocation(debug_program, matrix.as_ptr() as *const ffi::types::GLchar),
                uniform_tex_matrix: gl
                    .GetUniformLocation(debug_program, tex_matrix.as_ptr() as *const ffi::types::GLchar),
                uniform_alpha: gl
                    .GetUniformLocation(debug_program, alpha.as_ptr() as *const ffi::types::GLchar),
                attrib_vert: gl.GetAttribLocation(debug_program, vert.as_ptr() as *const ffi::types::GLchar),
                attrib_vert_position: gl
                    .GetAttribLocation(debug_program, vert_position.as_ptr() as *const ffi::types::GLchar),
                additional_uniforms: additional_uniforms
                    .iter()
                    .map(|uniform| {
                        let name = CString::new(uniform.name.as_bytes()).expect("Interior null in name");
                        let location =
                            gl.GetUniformLocation(debug_program, name.as_ptr() as *const ffi::types::GLchar);
                        (
                            uniform.name.clone().into_owned(),
                            UniformDesc {
                                location,
                                type_: uniform.type_,
                            },
                        )
                    })
                    .collect(),
            },
            // debug flags
            uniform_tint: gl.GetUniformLocation(debug_program, tint.as_ptr() as *const ffi::types::GLchar),
        })
    };

    Ok(GlesTexProgram(Arc::new(GlesTexProgramInner {
        variants: [
            create_variant(&[])?,
            create_variant(&[shaders::NO_ALPHA])?,
            create_variant(&[shaders::EXTERNAL])?,
        ],
        destruction_callback_sender,
    })))
}

pub(super) unsafe fn solid_program(gl: &ffi::Gles2) -> Result<GlesSolidProgram, GlesError> {
    let program = link_program(gl, shaders::VERTEX_SHADER_SOLID, shaders::FRAGMENT_SHADER_SOLID)?;

    let matrix = c"matrix";
    let color = c"color";
    let vert = c"vert";
    let position = c"position";

    Ok(GlesSolidProgram {
        program,
        uniform_matrix: gl.GetUniformLocation(program, matrix.as_ptr() as *const ffi::types::GLchar),
        uniform_color: gl.GetUniformLocation(program, color.as_ptr() as *const ffi::types::GLchar),
        attrib_vert: gl.GetAttribLocation(program, vert.as_ptr() as *const ffi::types::GLchar),
        attrib_position: gl.GetAttribLocation(program, position.as_ptr() as *const ffi::types::GLchar),
    })
}

#[cfg(test)]
mod tests {
    use super::{fragment_output, managed_fragment, names_extension};

    #[test]
    fn fragment_output_names_the_es3_color_output() {
        assert_eq!(fragment_output("precision mediump float;\nvoid main() {}"), None);
        assert_eq!(fragment_output("#version 100\nvoid main() {}"), None);
        assert_eq!(
            fragment_output("#version 300 es\nin vec2 uv;\nout vec4 fragColor;\nvoid main() {}"),
            Some("fragColor")
        );
        assert_eq!(
            fragment_output("#version 300 es\nlayout(location = 0) out highp vec4 color; // target\n"),
            Some("color")
        );
        assert_eq!(fragment_output("#version 300 es\nout float depth;\n"), None);
    }

    #[test]
    fn managed_fragment_wraps_main_with_the_color_conversion() {
        let color = include_str!("color.frag");
        // GLSL ES 1.00: the conversion reads and writes gl_FragColor.
        let es2 = "precision mediump float;\nvoid main() { gl_FragColor = vec4(1.0); }\n";
        let managed = managed_fragment(es2);
        assert_eq!(
            managed,
            format!("precision mediump float;\nvoid smithay_original_main() {{ gl_FragColor = vec4(1.0); }}\n{color}")
        );
        assert_eq!(managed.matches("void main(").count(), 1);

        // GLSL ES 3.00: the shader's own output replaces gl_FragColor.
        let es3 = "#version 300 es\nout highp vec4 fragColor;\nvoid main() { fragColor = vec4(1.0); }\n";
        let managed = managed_fragment(es3);
        assert!(managed.starts_with(
            "#version 300 es\nout highp vec4 fragColor;\nvoid smithay_original_main() { fragColor = vec4(1.0); }\n"
        ));
        assert!(managed.ends_with(&color.replace("gl_FragColor", "fragColor")));
        assert!(!managed.contains("gl_FragColor"));
        assert_eq!(managed.matches("void main(").count(), 1);

        // The cache entry is named by the source: equal input, equal output.
        assert_eq!(managed, managed_fragment(es3));
    }

    #[test]
    fn names_extension_matches_whole_names() {
        let list = "GL_OES_EGL_image GL_KHR_parallel_shader_compile GL_KHR_debug";
        assert!(names_extension(list, "GL_KHR_parallel_shader_compile"));
        assert!(names_extension(list, "GL_KHR_debug"));
        assert!(!names_extension(list, "GL_KHR_parallel_shader"));
        assert!(!names_extension(
            "GL_KHR_parallel_shader_compile_extra",
            "GL_KHR_parallel_shader_compile"
        ));
        assert!(!names_extension("", "GL_KHR_parallel_shader_compile"));
    }
}
