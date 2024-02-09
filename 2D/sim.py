import pymunk, math, json
import numpy as np

class Simulation:
    def __init__(self, C_R=0.9):
        # Simulation parameters
        self.FPS = 1000
        self.INV_FPS = 1/self.FPS
        self.NUM_STEPS = self.FPS*100
        self.BOUNCE_COUNT = 3 
        self.EPSILON = 1e-1
        self.THICKNESS = 0.2
        self.ball_radius = 0.2

        # Space setup
        self.space = pymunk.Space()
        self.space.gravity = (0, -9.81)
        self.C_R = C_R

        # Default initial conditions
        height = 10
        horizontal_velocity = 10

        # Ball setup
        moment_of_inertia = pymunk.moment_for_circle(1, 0, self.ball_radius, (0, 0))
        self.ball_body = pymunk.Body(1, moment_of_inertia)
        self.ball_body.position = (0, height)
        self.ball_body.velocity = (horizontal_velocity, 0) 
        self.ball_shape = pymunk.Circle(self.ball_body, self.ball_radius)
        self.ball_shape.elasticity = self.C_R
        self.space.add(self.ball_body, self.ball_shape)

        # Add box to prevent ball from going to infinity
        box_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        box_shape = pymunk.Segment(box_body, (-10000, -50), (10000, -50), self.THICKNESS)
        box_shape.elasticity = self.C_R
        self.space.add(box_body, box_shape)

        # Setting up collision handling
        handler = self.space.add_collision_handler(0, 0)
        handler.begin = self.collision_handler

        self.collisions = []

    # Function to detect and handle collisions
    def collision_handler(self, arbiter, space, data):
        contact_points = arbiter.contact_point_set
        for point in contact_points.points:
            x, _ = point.point_a
            if len(self.collisions)==0 or abs(self.collisions[-1]-x) > self.EPSILON:
                self.collisions.append(x)
        return True
    
    def get_bounces(self, height, horizontal_velocity):
        self.ball_body.position = (0, height)
        self.ball_body.velocity = (horizontal_velocity, 0) 
        self.collisions = []
        for _ in range(self.NUM_STEPS):
            self.space.step(self.INV_FPS)
            if len(self.collisions) >= self.BOUNCE_COUNT:
                break
        if len(self.collisions) < self.BOUNCE_COUNT:
            if isinstance(self, UnevenSimulation):
                print(f"Simulation failed to bounce for IC {height}, {horizontal_velocity}, for params A={self.A}, F={self.F}, P={self.P}")
            else:
                print(f"Simulation failed to bounce for IC {height}, {horizontal_velocity}")
            return [0, 0, 0]
        # if len(self.collisions) < self.BOUNCE_COUNT:
        #     self.collisions += [float("inf")]*(self.BOUNCE_COUNT-len(self.collisions))
        return self.collisions[:self.BOUNCE_COUNT]
    
    def get_bounces_as_string(self, height, horizontal_velocity):
        try:
            height = float(height)
            horizontal_velocity = float(horizontal_velocity)
        except Exception as e:
            raise ValueError(f"initial_conditions error, got {height}, {horizontal_velocity}, {e}")
        bounce_list = self.get_bounces(height, horizontal_velocity)
        return ", ".join([f"Bounce {i+1}: {bounce:.2f}cm" for i, bounce in enumerate(bounce_list)])

    def get_bounces_as_string_IC(self, IC):
        try:
            IC = IC[IC.index("{") : IC.rindex("}") + 1]
            j = json.loads(IC)
        except Exception as e:
            raise ValueError(f"initial_conditions error, got {IC}, {e}")
        height = j['height']
        horizontal_velocity = j['horizontal_velocity']
        bounce_list = self.get_bounces(height, horizontal_velocity)
        return ", ".join([f"Bounce {i+1}: {bounce:.2f}cm" for i, bounce in enumerate(bounce_list)])

    def visualise(self, h=5, v=5):
        import pygame, random
        
        print("------------------ Visualising ---------------------")
        print(f"Initial Conditions: h={h}, v={v}")

        self.ball_body.position = (0, h)
        self.ball_body.velocity = (v, 0) 
        self.collisions = []

        def draw_gradient_background(screen, top_color, bottom_color):
            """
            Draw a vertical gradient from top_color at the top of the screen to bottom_color at the bottom.
            """
            for y in range(screen.get_height()):
                # Calculate the blending factor
                blend_factor = y / screen.get_height()
                
                # Linearly interpolate between the top and bottom colors
                color = (
                    int(top_color[0] * (1 - blend_factor) + bottom_color[0] * blend_factor),
                    int(top_color[1] * (1 - blend_factor) + bottom_color[1] * blend_factor),
                    int(top_color[2] * (1 - blend_factor) + bottom_color[2] * blend_factor)
                )
                
                pygame.draw.line(screen, color, (0, y), (screen.get_width(), y))

        pygame.init()
        width, height = 800, 800
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Bouncing Ball Sim")
        clock = pygame.time.Clock()

        # Visualization loop
        running = True
        zoom_factor = 75  # Change this to zoom in/out
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                # If scroll wheel is used, zoom in/out
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 4:
                        zoom_factor *= 1.1
                    elif event.button == 5:
                        zoom_factor /= 1.1
                # If spacebar is pressed, reset the ball
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # h = random.uniform(1, 20)
                        # v = random.uniform(1, 20)
                        self.ball_body.position = (0, h)
                        self.ball_body.velocity = (v, 0) 

                        # print("-------------------------------------------")
                        # print(f"New values: h={h}, v={v}")

            draw_gradient_background(screen, (135, 206, 235), (200, 192, 203))  # Sky blue to light pink for demonstration
            
            # Follow the ball with the "camera"
            ball_x, ball_y = self.ball_body.position
            offset_x = -ball_x * zoom_factor + width / 2
            offset_y = -ball_y * zoom_factor + height / 2

            # Drawing the ground segments with sine wave
            for segment in self.space.shapes:
                if isinstance(segment, pymunk.Segment):
                    ax, ay = segment.a.x * zoom_factor + offset_x, segment.a.y * zoom_factor + offset_y
                    bx, by = segment.b.x * zoom_factor + offset_x, segment.b.y * zoom_factor + offset_y
                    pygame.draw.line(screen, (235, 20, 20), (ax, height - ay), (bx, height - by), int(zoom_factor*0.1))

            # Drawing the ball
            pygame.draw.circle(screen, (0, 0, 255), (int(ball_x * zoom_factor + offset_x), int(height - (ball_y * zoom_factor + offset_y))), int((self.ball_radius+0.1) * zoom_factor))

            pygame.display.flip()
            clock.tick(self.FPS/10)
            self.space.step(self.INV_FPS*10)

            if len(self.collisions)>0:
                print(f"Collisions: {self.collisions}")
                self.collisions = []
                
        self.collisions = []
        print("------------------ Done Visualising ---------------------")
        pygame.quit()

class CustomSimulation(Simulation):
    def __init__(self, FUNCTION):
        super().__init__()
        self.FUNCTION = FUNCTION
        N_ITER = 0.1
        LIMS = [-100, 250]
        for x in np.arange(LIMS[0], LIMS[1], N_ITER):
            x_coord_1 = x
            y_coord_1 = self.FUNCTION(x_coord_1)
            x_coord_2 = x + N_ITER
            y_coord_2 = self.FUNCTION(x_coord_2)
            segment = pymunk.Segment(self.space.static_body, (x_coord_1, y_coord_1), (x_coord_2, y_coord_2), self.THICKNESS)
            segment.elasticity = self.C_R
            self.space.add(segment)

class EvenSimulation(Simulation):
    def __init__(self):
        super().__init__()

        # Ground setup
        ground_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        ground_shape = pymunk.Segment(ground_body, (-100, 0), (500, 0), self.THICKNESS)  # Static ground at y=0
        ground_shape.body = ground_body  # Explicitly associating the shape with its body
        ground_shape.elasticity = self.C_R
        self.space.add(ground_body, ground_shape)

class UnevenSimulation(Simulation):
    def __init__(self, A=1, F=1, P=0, C_R=0.9):
        super().__init__(C_R=C_R)

        self.A = A
        self.F = F
        self.P = P

        self.set_surface(A, F, P)

    def set_surface(self, A=1, F=1, P=0):
        # Sine wave ground setup
        N_ITER = 0.2
        LIMS = [-150, 350]
        for x in np.arange(LIMS[0], LIMS[1], N_ITER):
            x_coord_1 = x
            y_coord_1 = A * math.sin(F * x_coord_1 + P)
            x_coord_2 = x + N_ITER
            y_coord_2 = A * math.sin(F * x_coord_2 + P)
            segment = pymunk.Segment(self.space.static_body, (x_coord_1, y_coord_1), (x_coord_2, y_coord_2), self.THICKNESS)
            segment.elasticity = self.C_R
            self.space.add(segment)

def visualise_even(h=None, v=None):
    import random

    SIMULATION = EvenSimulation

    sim = SIMULATION()

    if all([x is not None for x in [h, v]]):
        print("Using custom values")
    else:
        print("Using random values")
        h = random.uniform(1, 20)
        v = random.uniform(1, 20)

    print(f"Values: h={h}, v={v}")
    sim.visualise(h, v)

def visualise_uneven(h=None, v=None, A=None, F=None, P=None):
    import random

    SIMULATION = UnevenSimulation

    if all([x is not None for x in [h, v, A, F, P]]):
        print("Using custom values")
    else:
        print("Using random values")
        h = random.uniform(1, 20)
        v = random.uniform(1, 20)
        A = random.uniform(0.5, 5)
        F = random.uniform(0.1, 4)
        P = random.uniform(0.5, 4.5)

    print(f"Values: h={h}, v={v}, A={A}, F={F}, P={P}")
    sim = SIMULATION(A, F, P)
    sim.visualise(h, v)

def test_even_random_uniform(N):
    import random, time
    from tqdm import tqdm
    start_time = time.time()
    same_count = 0
    total_count = 0
    for i in tqdm(range(N), desc="Running Even Simulations"):
        sim = EvenSimulation()
        for _ in range(N):
            h = random.uniform(1, 20)
            v = random.uniform(1, 20)
            b = sim.get_bounces(h, v)
            assert len(b) == 3, f"Expected 3 bounces, got {b} at {h}, {v} in iteration {i}"
            if b[0] == b[1] == b[2]:
                same_count += 1
            total_count += 1
    end_time = time.time()
    print(f"Same bounces: {same_count}/{total_count}")
    print(f"Average time per simulation: {(end_time - start_time)/total_count} seconds")
    print(f"Passed test_even_random_uniform")

def test_uneven_random_uniform(N):
    import random, time
    from tqdm import tqdm
    start_time = time.time()
    same_count = 0
    total_count = 0
    for i in tqdm(range(N), desc="Running Uneven Simulations"):
        A = random.uniform(0.5, 5)
        F = random.uniform(0.1, 4)
        P = random.uniform(0.5, 4.5)
        sim = UnevenSimulation(A, F, P)
        for _ in range(N):
            h = random.uniform(5, 20)
            v = random.uniform(0, 20)
            b = sim.get_bounces(h, v)
            assert len(b) == 3, f"Expected 3 bounces, got {b} at {h}, {v} in iteration {i}"
            if b[0] == b[1] == b[2]:
                print(f"Bounces: {b}")
                print(f"{v}, {h}, {A}, {F}, {P}")
                visualise_uneven(h, v, A, F, P)
                same_count += 1
            total_count += 1
    end_time = time.time()
    print(f"Same bounces: {same_count}/{total_count}")
    print(f"Average time per simulation: {(end_time - start_time)/total_count} seconds")
    print(f"Passed test_uneven_random_uniform")

if __name__ == "__main__":
    N = 100
    test_even_random_uniform(N)
    test_uneven_random_uniform(N)
    #visualise()


