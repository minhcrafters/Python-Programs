import pygame


def scale_image(image: pygame.surface.Surface, factor: float) -> pygame.surface.Surface:
    return pygame.transform.scale(
        image, (image.get_width() * factor, image.get_height() * factor)
    )


def draw_text(
    screen: pygame.surface.Surface,
    font: pygame.font.Font,
    text: str,
    pos: tuple[float | int, float | int],
    colour: tuple[int, int, int] = (255, 255, 255),
    drop_colour: tuple[int, int, int, int] = (0, 0, 0, 128),
    anti_aliasing: bool = False,
    anchor: str = "topleft",
    shadow: bool = False,
    shadow_offset: float = 1,
    fade_out_when_collided: bool = False,
    collision_rect: pygame.Rect | None = None,
) -> pygame.surface.Surface:
    if shadow:
        dropshadow_offset = shadow_offset + (
            font.size(text)[0] // (font.size(text)[0] / 1.5)
        )

        text_bitmap = font.render(text, anti_aliasing, drop_colour).convert_alpha()
        rect = text_bitmap.get_rect()
        setattr(rect, anchor, pos)

        if fade_out_when_collided and rect.colliderect(collision_rect):
            text_bitmap.set_alpha(128 - drop_colour[3])
        else:
            text_bitmap.set_alpha(255 - drop_colour[3])

        screen.blit(
            text_bitmap, (rect.x + dropshadow_offset, rect.y + dropshadow_offset)
        )

    text_bitmap = font.render(text, anti_aliasing, colour)
    rect = text_bitmap.get_rect()
    setattr(rect, anchor, pos)

    if fade_out_when_collided and rect.colliderect(collision_rect):
        text_bitmap.set_alpha(128)
    else:
        text_bitmap.set_alpha(255)

    screen.blit(text_bitmap, rect)
    return text_bitmap
