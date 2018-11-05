from tkinter import *
from threading import Thread

class SolutionCanvas():

    def __init__(self, width, height, room_width, room_height, fixed_shapes):
        self.width = width
        self.height = height
        self.room_width = room_width
        self.room_height = room_height
        self.canvas = None
        self.fixed_shapes = fixed_shapes
        self.polys = []
        self.partials = []
        self.scalar = min(self.width / self.room_width, self.height / self.room_height)

        thread = Thread(target=self._mainloop)
        thread.start()

    def _mainloop(self):
        master = Tk()
        self.canvas = Canvas(master, width=self.width, height=self.height)
        self.canvas.pack()
        self.canvas.create_rectangle(0, 0, self.room_width * self.scalar, self.room_height * self.scalar, fill="black")
        for shape in self.fixed_shapes:
            self._add_to_canvas(shape, self.scalar, "white")
        master.mainloop()

    def _add_to_canvas(self, shape, scalar, color):
        x, y = shape.exterior.coords.xy
        return self.canvas.create_polygon([(x[i] * scalar, y[i] * scalar) for i in range(len(x))], fill=color)

    def paint_partial(self, shapes):
        for poly in self.partials:
            self.canvas.delete(poly)
        self.partials = []
        for shape in shapes:
            self.partials.append(self._add_to_canvas(shape, self.scalar, "red"))
        self.canvas.update_idletasks()

    def paint_solution(self, solution_shapes):
        for poly in self.polys:
            self.canvas.delete(poly)
        self.polys = []
        for shape in solution_shapes:
            self.polys.append(self._add_to_canvas(shape, self.scalar, "green"))
        self.canvas.update_idletasks()

