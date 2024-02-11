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
) -> pygame.surface.Surface:
    if shadow:
        dropshadow_offset = shadow_offset + (
            font.size(text)[0] // (font.size(text)[0] / 1.5)
        )

        text_bitmap = font.render(text, anti_aliasing, drop_colour).convert_alpha()
        text_bitmap.set_alpha(drop_colour[3])
        rect = text_bitmap.get_rect()
        setattr(rect, anchor, pos)

        screen.blit(
            text_bitmap, (rect.x + dropshadow_offset, rect.y + dropshadow_offset)
        )

    text_bitmap = font.render(text, anti_aliasing, colour)
    rect = text_bitmap.get_rect()
    setattr(rect, anchor, pos)
    screen.blit(text_bitmap, rect)
    return text_bitmap
