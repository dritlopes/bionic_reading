# Inspired by vis.py from EyeKit

class Image:

    def __init__(self, screen_width, screen_height, font_face, font_size):

        self.screen_width = int(screen_width)
        self.screen_height = int(screen_height)
        self.font_face = font_face
        self.font_size = font_size
        self._background_color = (1, 1, 1)
        self._components = []

    def _add_component(self, func, arguments):
        """
        Add a component to the stack. This should be a draw_ function and its
        arguments. This function will be called with its arguments at save
        time.
        """
        self._components.append((func, arguments))

def visualise_fixations_on_text(trial):

    for ia in trial.text.ias:
        pass