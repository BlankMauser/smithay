use std::{mem, ops::Range};

use glam::{Affine2, Vec2};

use super::{Draw, DrawKind, Vertex, WgpuError, WgpuFrame, WgpuTarget, WgpuTexture, texture::has_alpha};
use crate::{
    backend::renderer::{Blit, BlitFrame, Color32F, ContextId, DebugFlags, Frame, Texture, TextureFilter},
    utils::{Buffer, Physical, Point, Rectangle, Size, Transform},
};

impl WgpuFrame<'_> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn render_texture(
        &mut self,
        texture: &WgpuTexture,
        src: Rectangle<f64, Buffer>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        src_transform: Transform,
        alpha: f32,
        blit: bool,
    ) -> Result<(), WgpuError> {
        if texture.context_id() != &self.renderer.context_id {
            return Err(WgpuError::ForeignTexture);
        }
        if texture.same_storage(&self.target) {
            return Err(WgpuError::Unsupported);
        }
        if !texture
            .raw()
            .usage()
            .contains(::wgpu::TextureUsages::TEXTURE_BINDING)
        {
            return Err(WgpuError::UnsupportedTextureUsage);
        }
        if src.size.is_empty() || dst.size.is_empty() || texture.size().is_empty() {
            return Ok(());
        }
        let source_right = src.loc.x + src.size.w;
        let source_bottom = src.loc.y + src.size.h;
        if !src.loc.x.is_finite()
            || !src.loc.y.is_finite()
            || !source_right.is_finite()
            || !source_bottom.is_finite()
            || src.loc.x < 0.0
            || src.loc.y < 0.0
            || source_right > texture.width() as f64
            || source_bottom > texture.height() as f64
        {
            return Err(WgpuError::InvalidRegion);
        }

        let texture_size = texture.size();
        let transformed_src_size = src_transform.transform_size(src.size);
        let scale = transformed_src_size.to_f64() / dst.size.to_f64();
        let mut texture_matrix = Affine2::from_scale(Vec2::new(scale.x as f32, scale.y as f32));
        let translation = match src_transform {
            Transform::Normal => Affine2::IDENTITY,
            Transform::_90 => Affine2::from_translation(Vec2::new(0.0, transformed_src_size.w as f32)),
            Transform::_180 => Affine2::from_translation(Vec2::new(
                transformed_src_size.w as f32,
                transformed_src_size.h as f32,
            )),
            Transform::_270 => Affine2::from_translation(Vec2::new(transformed_src_size.h as f32, 0.0)),
            Transform::Flipped => Affine2::from_translation(Vec2::new(transformed_src_size.w as f32, 0.0)),
            Transform::Flipped90 => Affine2::IDENTITY,
            Transform::Flipped180 => Affine2::from_translation(Vec2::new(0.0, transformed_src_size.h as f32)),
            Transform::Flipped270 => Affine2::from_translation(Vec2::new(
                transformed_src_size.h as f32,
                transformed_src_size.w as f32,
            )),
        };
        texture_matrix = src_transform.matrix() * texture_matrix;
        texture_matrix = translation * texture_matrix;
        texture_matrix =
            Affine2::from_translation(Vec2::new(src.loc.x as f32, src.loc.y as f32)) * texture_matrix;
        texture_matrix = Affine2::from_scale(Vec2::new(
            1.0 / texture_size.w as f32,
            1.0 / texture_size.h as f32,
        )) * texture_matrix;

        let start = self.vertices.len() as u32;
        let mut tint = [alpha; 4];
        if !blit && self.renderer.debug_flags.contains(DebugFlags::TINT) {
            tint[1] *= 0.5;
            tint[2] *= 0.5;
        }
        for damage in damage {
            let Some((local_rect, rect)) = Self::damage_rect(dst, *damage)? else {
                continue;
            };
            let local = local_rect.loc;
            let x0 = local.x as f32;
            let y0 = local.y as f32;
            let x1 = local
                .x
                .checked_add(local_rect.size.w)
                .ok_or(WgpuError::InvalidRegion)? as f32;
            let y1 = local
                .y
                .checked_add(local_rect.size.h)
                .ok_or(WgpuError::InvalidRegion)? as f32;
            let mut coords = [
                texture_matrix.transform_point2(Vec2::new(x0, y0)).to_array(),
                texture_matrix.transform_point2(Vec2::new(x1, y0)).to_array(),
                texture_matrix.transform_point2(Vec2::new(x0, y1)).to_array(),
                texture_matrix.transform_point2(Vec2::new(x1, y1)).to_array(),
            ];
            if !blit && texture.flipped() {
                for coord in &mut coords {
                    coord[1] = 1.0 - coord[1];
                }
            }
            self.push_quad(rect, coords, tint, !has_alpha(texture.format().unwrap()))?;
        }
        let end = self.vertices.len() as u32;
        if start != end {
            self.draws.push(Draw {
                vertices: start..end,
                kind: DrawKind::Texture {
                    texture: texture.clone(),
                    opaque: blit || (!has_alpha(texture.format().unwrap()) && alpha == 1.0),
                },
            });
        }
        Ok(())
    }

    fn push_quad(
        &mut self,
        rect: Rectangle<i32, Physical>,
        tex_coords: [[f32; 2]; 4],
        color: [f32; 4],
        force_opaque: bool,
    ) -> Result<Range<u32>, WgpuError> {
        if self.vertices.len() > u32::MAX as usize - 6 {
            return Err(WgpuError::Unsupported);
        }
        let start = self.vertices.len() as u32;
        let right = rect
            .loc
            .x
            .checked_add(rect.size.w)
            .ok_or(WgpuError::InvalidRegion)?;
        let bottom = rect
            .loc
            .y
            .checked_add(rect.size.h)
            .ok_or(WgpuError::InvalidRegion)?;
        let corners = [
            Point::from((rect.loc.x, rect.loc.y)),
            Point::from((right, rect.loc.y)),
            Point::from((rect.loc.x, bottom)),
            Point::from((right, bottom)),
        ];
        let indices = [0, 2, 1, 1, 2, 3];
        let target_size = self.transform.transform_size(self.output_size);
        for index in indices {
            let point = self
                .transform
                .transform_point_in(corners[index], &self.output_size);
            let x = point.x as f32 / target_size.w as f32 * 2.0 - 1.0;
            let y = 1.0 - point.y as f32 / target_size.h as f32 * 2.0;
            self.vertices.push(Vertex {
                position: [x, y],
                tex_coord: tex_coords[index],
                color,
                force_opaque: force_opaque as u32 as f32,
            });
        }
        Ok(start..self.vertices.len() as u32)
    }

    fn damage_rect(
        dst: Rectangle<i32, Physical>,
        damage: Rectangle<i32, Physical>,
    ) -> Result<Option<(Rectangle<i32, Physical>, Rectangle<i32, Physical>)>, WgpuError> {
        for rect in [dst, damage] {
            if rect.loc.x.checked_add(rect.size.w).is_none() || rect.loc.y.checked_add(rect.size.h).is_none()
            {
                return Err(WgpuError::InvalidRegion);
            }
        }
        let Some(local) = damage.intersection(Rectangle::from_size(dst.size)) else {
            return Ok(None);
        };
        let x = dst
            .loc
            .x
            .checked_add(local.loc.x)
            .ok_or(WgpuError::InvalidRegion)?;
        let y = dst
            .loc
            .y
            .checked_add(local.loc.y)
            .ok_or(WgpuError::InvalidRegion)?;
        Ok(Some((local, Rectangle::new((x, y).into(), local.size))))
    }

    fn solid(
        &mut self,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        color: Color32F,
        replace: bool,
    ) -> Result<(), WgpuError> {
        let start = self.vertices.len() as u32;
        for damage in damage {
            let Some((_, rect)) = Self::damage_rect(dst, *damage)? else {
                continue;
            };
            self.push_quad(rect, [[0.0; 2]; 4], color.components(), true)?;
        }
        let end = self.vertices.len() as u32;
        if start != end {
            self.draws.push(Draw {
                vertices: start..end,
                kind: if replace {
                    DrawKind::Replace
                } else {
                    DrawKind::Solid
                },
            });
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<crate::backend::renderer::sync::SyncPoint, WgpuError> {
        let result = self.renderer.submit(
            &self.target,
            self.transform.transform_size(self.output_size),
            &self.vertices,
            &self.draws,
        );
        self.vertices.clear();
        self.draws.clear();
        result
    }

    fn finish_internal(mut self) -> Result<crate::backend::renderer::sync::SyncPoint, WgpuError> {
        let result = self.flush();
        self.renderer.vertices = mem::take(&mut self.vertices);
        self.renderer.vertices.clear();
        self.renderer.draws = mem::take(&mut self.draws);
        self.renderer.draws.clear();
        result
    }
}

impl Frame for WgpuFrame<'_> {
    type Error = WgpuError;
    type TextureId = WgpuTexture;

    fn context_id(&self) -> ContextId<WgpuTexture> {
        self.renderer.context_id.clone()
    }

    fn clear(&mut self, color: Color32F, at: &[Rectangle<i32, Physical>]) -> Result<(), Self::Error> {
        self.solid(Rectangle::from_size(self.output_size), at, color, true)
    }

    fn draw_solid(
        &mut self,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        color: Color32F,
    ) -> Result<(), Self::Error> {
        self.solid(dst, damage, color, color.is_opaque())
    }

    fn render_texture_from_to(
        &mut self,
        texture: &WgpuTexture,
        src: Rectangle<f64, Buffer>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        _opaque_regions: &[Rectangle<i32, Physical>],
        src_transform: Transform,
        alpha: f32,
    ) -> Result<(), Self::Error> {
        self.render_texture(texture, src, dst, damage, src_transform, alpha, false)
    }

    fn transformation(&self) -> Transform {
        self.transform
    }

    fn output_size(&self) -> Size<i32, Physical> {
        self.output_size
    }

    fn wait(&mut self, sync: &crate::backend::renderer::sync::SyncPoint) -> Result<(), Self::Error> {
        super::sync::wait(sync)
    }

    fn finish(self) -> Result<crate::backend::renderer::sync::SyncPoint, Self::Error> {
        self.finish_internal()
    }
}

impl BlitFrame<WgpuTarget> for WgpuFrame<'_> {
    fn blit_to(
        &mut self,
        to: &mut WgpuTarget,
        src: Rectangle<i32, Physical>,
        dst: Rectangle<i32, Physical>,
        filter: TextureFilter,
    ) -> Result<crate::backend::renderer::sync::SyncPoint, WgpuError> {
        let _ = self.flush()?;
        let from = WgpuTarget {
            texture: self.target.clone(),
        };
        self.renderer.blit(&from, to, src, dst, filter)
    }

    fn blit_from(
        &mut self,
        from: &WgpuTarget,
        src: Rectangle<i32, Physical>,
        dst: Rectangle<i32, Physical>,
        filter: TextureFilter,
    ) -> Result<crate::backend::renderer::sync::SyncPoint, WgpuError> {
        let _ = self.flush()?;
        let mut to = WgpuTarget {
            texture: self.target.clone(),
        };
        self.renderer.blit(from, &mut to, src, dst, filter)
    }
}
