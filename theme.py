from gradio.themes.base import Base


class CustomTheme(Base):

    def __init__(self):
        super().__init__()

        white = "#FFFFFF"
        purple = "#571DF9"
        red = "#FA5558"

        body_background_fill = "repeating-linear-gradient(45deg, rgba(39, 24, 152, 0.8), rgba(39, 24, 152, 0.8) 10px, rgba(39, 24, 152, 1.0), rgba(250, 85, 88, 1.0) 20px);"

        super().set(
            body_background_fill=body_background_fill,
            button_primary_background_fill=red,
            button_primary_text_color=white,
            button_secondary_background_fill=red,
            button_secondary_text_color=white,
            button_secondary_border_color=white,
            color_accent_soft="rgb(72, 160, 203)",
            border_color_accent_subdued="rgb(72, 160, 203)",
        )
