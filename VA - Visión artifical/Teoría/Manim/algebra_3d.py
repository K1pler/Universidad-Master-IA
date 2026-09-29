"""Escenas Manim para visualizar conceptos de álgebra lineal en 3D.

Sin MathTex/LaTeX: usa Text para evitar dependencias bloqueadas en Windows.
"""

from manim import *


class VectoresYPlano(ThreeDScene):
    """Dos vectores generan un plano; el producto cruzado es la normal."""

    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)

        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[-2, 3, 1],
            x_length=6,
            y_length=6,
            z_length=4,
        )
        # Etiquetas sin LaTeX
        x_lab = Text("x", font_size=24).next_to(axes.x_axis.get_end(), RIGHT)
        y_lab = Text("y", font_size=24).next_to(axes.y_axis.get_end(), UP)
        z_lab = Text("z", font_size=24).next_to(axes.z_axis.get_end(), OUT)
        self.add_fixed_orientation_mobjects(x_lab, y_lab, z_lab)

        v1_coords = np.array([2.0, 0.5, 0.5])
        v2_coords = np.array([0.5, 2.0, 0.8])
        normal = np.cross(v1_coords, v2_coords)

        v1 = Arrow3D(ORIGIN, v1_coords, color=BLUE, thickness=0.03)
        v2 = Arrow3D(ORIGIN, v2_coords, color=GREEN, thickness=0.03)
        n_arrow = Arrow3D(
            ORIGIN,
            normal / np.linalg.norm(normal) * 1.8,
            color=RED,
            thickness=0.03,
        )

        v1_label = Text("v1", font_size=22, color=BLUE)
        v2_label = Text("v2", font_size=22, color=GREEN)
        n_label = Text("n = v1 x v2", font_size=20, color=RED)
        self.add_fixed_orientation_mobjects(v1_label, v2_label, n_label)
        v1_label.move_to(v1_coords + np.array([0.35, 0, 0.2]))
        v2_label.move_to(v2_coords + np.array([0, 0.35, 0.2]))
        n_label.move_to(normal / np.linalg.norm(normal) * 1.8 + np.array([0.3, 0.3, 0.2]))

        plane = Surface(
            lambda u, v: u * v1_coords + v * v2_coords,
            u_range=[-0.6, 1.1],
            v_range=[-0.6, 1.1],
            resolution=(12, 12),
        )
        plane.set_style(fill_opacity=0.45, stroke_width=0.5, fill_color=PURPLE)

        title = Text("Span{v1, v2} = plano", font_size=26)
        title.to_corner(UL)
        self.add_fixed_in_frame_mobjects(title)

        self.play(Create(axes), FadeIn(x_lab), FadeIn(y_lab), FadeIn(z_lab))
        self.play(Create(v1), FadeIn(v1_label))
        self.play(Create(v2), FadeIn(v2_label))
        self.play(Create(plane))
        self.play(Create(n_arrow), FadeIn(n_label))
        self.begin_ambient_camera_rotation(rate=0.15)
        self.wait(4)
        self.stop_ambient_camera_rotation()
        self.wait(0.5)


class PlanoEcuacion(ThreeDScene):
    """Plano ax + by + cz = d y su vector normal."""

    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=30 * DEGREES)

        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[-2, 3, 1],
            x_length=6,
            y_length=6,
            z_length=4,
        )

        # Plano: x + y + z = 2  =>  z = 2 - x - y
        a, b, c, d = 1.0, 1.0, 1.0, 2.0
        plane = Surface(
            lambda x, y: np.array([x, y, (d - a * x - b * y) / c]),
            u_range=[-2, 2],
            v_range=[-2, 2],
            resolution=(16, 16),
        )
        plane.set_style(fill_opacity=0.5, stroke_width=0.4, fill_color=TEAL)

        normal = np.array([a, b, c])
        normal_u = normal / np.linalg.norm(normal)
        point_on_plane = np.array([0.0, 0.0, d / c])
        n_arrow = Arrow3D(
            point_on_plane,
            point_on_plane + normal_u * 1.5,
            color=YELLOW,
            thickness=0.035,
        )

        eq = Text("x + y + z = 2", font_size=26, color=TEAL).to_corner(UL)
        n_tex = Text("n = (1, 1, 1)", font_size=22, color=YELLOW).next_to(
            eq, DOWN, aligned_edge=LEFT
        )
        self.add_fixed_in_frame_mobjects(eq, n_tex)

        self.play(Create(axes))
        self.play(Create(plane), Write(eq))
        self.play(Create(n_arrow), Write(n_tex))
        self.begin_ambient_camera_rotation(rate=0.12)
        self.wait(4)
        self.stop_ambient_camera_rotation()
        self.wait(0.5)
