
                if HAS_CAIROSVG:
                    png_bytes = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'))
                    png_file = io.BytesIO(png_bytes)

                    surface = pygame.image.load(png_file)
                    surface = surface.convert_alpha()

                else:
                    # fallback to unicode
                    surface = self._create_unicode_piece(piece_type, color, piece_size)
                