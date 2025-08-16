import pygame
import random
import math
import time
import sys
import json
import uuid
import numpy as np
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
from perlin_noise import PerlinNoise
import heapq
from typing import List, Dict, Tuple, Optional, Set

# Initialize pygame
pygame.init()
WIDTH, HEIGHT = 1400, 900
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("The Genesis Machine: Advanced Digital Life Universe")
clock = pygame.time.Clock()

# Colors
BACKGROUND = (8, 12, 25)
TEXT_COLOR = (180, 220, 255) 
HIGHLIGHT = (0, 200, 255)
FOOD_COLOR = (50, 200, 80)
WATER_COLOR = (40, 120, 220)
MOUNTAIN_COLOR = (100, 90, 80)
FOREST_COLOR = (30, 120, 50)
DESERT_COLOR = (210, 190, 130)
TUNDRA_COLOR = (220, 240, 255)
PLAINS_COLOR = (110, 170, 90)
SETTLEMENT_COLOR = (200, 150, 100)
WEATHER_RAIN = (100, 150, 200, 100) 
WEATHER_STORM = (80, 80, 120, 120) 

# Fonts
font_small = pygame.font.SysFont('Arial', 12)
font_medium = pygame.font.SysFont('Arial', 16)
font_large = pygame.font.SysFont('Arial', 24)
font_title = pygame.font.SysFont('Arial', 32, bold=True)

# Global simulation parameters
simulation_speed = 1.0
paused = False
show_debug = False
show_language = False
show_mating = False
show_dreamers = False
show_conflicts = False
show_territories = False
show_disease = False
show_history = False
show_paths = False
show_debug_overlay = False 
show_console = False 
console_input = "" 
weather_state = "clear" 
weather_timer = 0 
weather_duration = 300 

# Camera and Zoom
camera_x, camera_y = 0, 0
zoom = 1.0
brightness = 1.0
selected_organism = None 
last_autosave_time = time.time() 

# Biome types
class Biome(Enum):
    OCEAN = 0
    COAST = 1
    PLAINS = 2
    FOREST = 3
    MOUNTAIN = 4
    DESERT = 5
    TUNDRA = 6
    RIVER = 7

# Evolution phases
class Phase(Enum):
    BLOB = 0
    HUNTER_GATHERER = 1
    EMOTION = 2
    DREAM = 3
    CANNIBAL = 4
    TRIBAL = 5
    LANGUAGE = 6
    CONFLICT = 7
    DISEASE = 8
    CIVILIZATION = 9

# Disease types
class DiseaseType(Enum):
    VIRUS = 0
    BACTERIA = 1
    FUNGAL = 2
    PRION = 3
    PARASITE = 4

# Roles in civilization
class Role(Enum):
    LEADER = 0
    PRIEST = 1
    WARRIOR = 2
    GATHERER = 3
    BUILDER = 4
    HEALER = 5

# Structure types
class StructureType(Enum):
    CAMP = 0
    HUT = 1
    TEMPLE = 2
    WATCHTOWER = 3
    STORAGE = 4

# Historical events
class EventType(Enum):
    BIRTH = 0
    DEATH = 1
    REPRODUCTION = 2
    DISCOVERY = 3
    DREAM = 4
    BELIEF_FORMED = 5
    TRIBE_FORMED = 6
    WAR = 7
    DISEASE_OUTBREAK = 8
    SETTLEMENT_BUILT = 9
    MYTH_CREATED = 10
    LANGUAGE_DEVELOPED = 11

# World generation
terrain_noise = PerlinNoise(octaves=8, seed=random.randint(0, 1000))
temperature_noise = PerlinNoise(octaves=4, seed=random.randint(0, 1000))
moisture_noise = PerlinNoise(octaves=4, seed=random.randint(0, 1000))

# Generate terrain
def generate_terrain():
    elevation = np.zeros((WIDTH//10, HEIGHT//10))
    temperature = np.zeros((WIDTH//10, HEIGHT//10))
    moisture = np.zeros((WIDTH//10, HEIGHT//10))

    for x in range(WIDTH//10):
        for y in range(HEIGHT//10):
            elevation[x][y] = terrain_noise([x/40, y/40])
            # Temperature gradient (colder at poles) 
            temperature[x][y] = temperature_noise([x/80, y/80]) - abs(y - HEIGHT//20) / (HEIGHT//10)
            moisture[x][y] = moisture_noise([x/60, y/60])

    return elevation, temperature, moisture

elevation, temperature, moisture = generate_terrain()

# Current evolution phase
current_phase = Phase.BLOB

# Historical records
historical_events = []

@dataclass
class HistoricalEvent:
    time: float
    event_type: EventType
    description: str
    location: Tuple[float, float]
    involved: List = field(default_factory=list)

    def __lt__(self, other):
        return self.time < other.time

@dataclass
class Disease:
    disease_id: str
    disease_type: DiseaseType
    transmission_rate: float
    mortality_rate: float
    mutation_rate: float
    immunity_resistance: float
    symptoms: List[str]
    color: Tuple[int, int, int]
    incubation_period: int = 0
    duration: int = 500

    def mutate(self):
        if random.random() < self.mutation_rate:
            self.transmission_rate = min(1.0, self.transmission_rate * random.uniform(0.8, 1.2))
            self.mortality_rate = min(1.0, self.mortality_rate * random.uniform(0.8, 1.2))
            self.immunity_resistance = min(1.0, self.immunity_resistance * random.uniform(0.8, 1.2))

            # Chance to change symptom 
            if random.random() < 0.2:
                self.symptoms.append(random.choice(["coughing", "bloating", "skin lesions", "blindness", "paralysis", "rage"]))

            # Rare type change 
            if random.random() < 0.005:
                self.disease_type = random.choice(list(DiseaseType))

@dataclass
class DNA:
    color: tuple = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
    size: float = random.uniform(8, 18)
    speed: float = random.uniform(0.7, 3.5)
    aggression: float = random.uniform(0.0, 1.0)
    sociability: float = random.uniform(0.0, 1.0)
    intelligence: float = random.uniform(0.0, 1.0)
    dream_recall: float = random.uniform(0.0, 0.15)
    vision_range: int = random.randint(60, 180)
    metabolism: float = random.uniform(0.008, 0.04) # Slower metabolism
    limb_count: int = random.randint(2, 8)
    shape: str = random.choice(['circle', 'triangle', 'pentagon'])
    immunity: float = random.uniform(0.0, 0.6)
    mutation_rate: float = random.uniform(0.05, 0.3)
    disease_resistance: Dict[str, float] = field(default_factory=dict)
    preferred_biome: Biome = random.choice(list(Biome))

@dataclass
class Emotion:
    fear: float = 0.0
    rage: float = 0.0
    hunger: float = 0.0
    sorrow: float = 0.0
    joy: float = 0.0
    curiosity: float = 0.0
    loyalty: float = 0.0

@dataclass
class Memory:
    events: deque = field(default_factory=lambda: deque(maxlen=30))
    traumas: list = field(default_factory=list)
    dreams: list = field(default_factory=list)

@dataclass
class Belief:
    origin_story: str = ""
    deities: list = field(default_factory=list)
    rituals: list = field(default_factory=list)
    taboos: list = field(default_factory=list)
    myths: list = field(default_factory=list)

@dataclass
class Tribe:
    tribe_id: str
    members: list = field(default_factory=list)
    color: tuple = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
    territory: list = field(default_factory=list)
    enemies: list = field(default_factory=list)
    allies: list = field(default_factory=list)
    language: dict = field(default_factory=dict)
    shared_beliefs: Belief = field(default_factory=Belief)
    settlements: list = field(default_factory=list)
    leader: object = None
    last_war_time: float = 0.0

@dataclass
class Structure:
    structure_id: str
    structure_type: StructureType
    x: float
    y: float
    size: float
    health: float = 100.0
    owner: object = None
    occupants: list = field(default_factory=list)

@dataclass
class Settlement:
    settlement_id: str
    x: float
    y: float
    radius: float
    tribe: object
    structures: list = field(default_factory=list)
    population: int = 0
    food_storage: float = 0.0

class Organism:
    def __init__(self, x, y, dna=None, parents=None):
        self.id = uuid.uuid4().hex[:8]
        self.x = x
        self.y = y
        self.dna = dna if dna else DNA()
        self.energy = 100.0
        self.age = 0
        self.max_age = random.randint(2000, 5000) # Longer lifespan
        self.emotion = Emotion()
        self.memory = Memory()
        self.target = None
        self.tribe = None
        self.language = {}
        self.belief = Belief()
        self.disease = None
        self.dreaming = False
        self.dream_start_time = 0
        self.dream_duration = 0
        self.role = None
        self.home = None
        self.path = []
        self.last_reproduction_time = 0
        self.immunities = {}
        self.symptoms = []
        self.disease_start_time = 0
        self.rect = pygame.Rect(x - self.dna.size, y - self.dna.size, self.dna.size * 2, self.dna.size * 2) 
        self.glyph = self.generate_glyph() 
        if parents:
            self.inherit_from_parents(parents)
            self.mutate()
        if current_phase.value >= Phase.TRIBAL.value:
            self.form_basic_beliefs()

    def inherit_from_parents(self, parents):
        # Inherit traits from parents 
        self.dna.color = random.choice(parents).dna.color
        self.dna.size = np.mean([p.dna.size for p in parents])
        self.dna.speed = np.mean([p.dna.speed for p in parents])
        self.dna.aggression = np.mean([p.dna.aggression for p in parents])
        self.dna.sociability = np.mean([p.dna.sociability for p in parents])
        self.dna.intelligence = np.mean([p.dna.intelligence for p in parents])
        self.dna.dream_recall = np.mean([p.dna.dream_recall for p in parents])
        self.dna.immunity = np.mean([p.dna.immunity for p in parents])
        self.dna.preferred_biome = random.choice(parents).dna.preferred_biome

        # Inherit partial immunities 
        for p in parents:
            for disease_id, immunity in p.immunities.items():
                self.immunities[disease_id] = immunity * random.uniform(0.7, 0.9)

        # Chance of trait mixing
        if random.random() < 0.4:
            self.dna.vision_range = random.choice(parents).dna.vision_range
        if random.random() < 0.4:
            self.dna.metabolism = random.choice(parents).dna.metabolism
        if random.random() < 0.4:
            self.dna.limb_count = random.choice(parents).dna.limb_count
        if random.random() < 0.4: 
            self.dna.shape = random.choice(parents).dna.shape

    def mutate(self):
        # Structural mutations 
        if random.random() < self.dna.mutation_rate:
            self.dna.size *= random.uniform(0.7, 1.3)
        if random.random() < self.dna.mutation_rate:
            self.dna.color = (
                max(0, min(255, self.dna.color[0] + random.randint(-40, 40))),
                max(0, min(255, self.dna.color[1] + random.randint(-40, 40))),
                max(0, min(255, self.dna.color[2] + random.randint(-40, 40)))
            )
        if random.random() < self.dna.mutation_rate / 2:
            self.dna.limb_count = max(2, self.dna.limb_count + random.choice([-1, 0, 1]))
            if self.dna.limb_count > 8:
                self.dna.limb_count = 8

        # Behavioral mutations 
        if random.random() < self.dna.mutation_rate:
            self.dna.aggression = max(0, min(1, self.dna.aggression + random.uniform(-0.3, 0.3)))
        if random.random() < self.dna.mutation_rate:
            self.dna.sociability = max(0, min(1, self.dna.sociability + random.uniform(-0.3, 0.3)))

        # Cognitive mutations 
        if random.random() < self.dna.mutation_rate / 3:
            self.dna.dream_recall = max(0, min(1, self.dna.dream_recall + random.uniform(-0.15, 0.15)))

        # Disease resistance mutation 
        if random.random() < self.dna.mutation_rate / 4:
            self.dna.immunity = max(0, min(1, self.dna.immunity + random.uniform(-0.1, 0.1)))

        # Biome preference mutation 
        if random.random() < 0.01:
            self.dna.preferred_biome = random.choice(list(Biome))

        # Rare trait emergence 
        if random.random() < 0.01:
            self.dna.dream_recall += 0.4
        if random.random() < 0.008 and current_phase.value >= Phase.LANGUAGE.value:
            self.dna.intelligence += 0.5
        if random.random() < 0.02: 
            self.dna.shape = random.choice(['circle', 'triangle', 'pentagon', 'square'])
        # print(f"Organism {self.id} mutated: size={self.dna.size:.2f}, shape={self.dna.shape}, aggression={self.dna.aggression:.2f}, sociability={self.dna.sociability:.2f}") 

    def generate_glyph(self):
        # Create a simple visual glyph for the organism 
        glyph = []
        elements = ["△", "◯", "□", "◇", "☆", "☾", "☀", "☁", "⚡"]
        for _ in range(3):
            glyph.append(random.choice(elements))
        return "".join(glyph)

    def form_basic_beliefs(self):
        # Form basic origin beliefs
        events = [e for e in self.memory.events if e[0] in ["birth", "death", "trauma"]]
        if events:
            event = random.choice(events)
            if event[0] == "birth":
                self.belief.origin_story = f"We were created by the {random.choice(['Sky', 'Earth', 'Water', 'Star', 'Wind'])} Spirits"
            elif event[0] == "death":
                self.belief.origin_story = f"We emerged from the {random.choice(['Great Darkness', 'Eternal Light', 'Primordial Sea', 'Silent Void'])}"
            elif event[0] == "trauma":
                self.belief.origin_story = f"The {random.choice(['Pain', 'Sorrow', 'Fear', 'Hunger'])} shaped our beginning"

        # Create a deity 
        deity_name = f"{random.choice(['Great', 'Eternal', 'Silent', 'Hidden', 'All-Seeing'])} {random.choice(['One', 'Spirit', 'Being', 'Presence'])}"
        deity_power = random.choice(["creation", "destruction", "life", "death", "dreams", "storms", "harvest"])
        self.belief.deities.append(f"{deity_name} of {deity_power}")

        # Create a taboo 
        self.belief.taboos.append(f"Do not {random.choice(['eat', 'touch', 'speak of', 'look at'])} the {random.choice(['red', 'blue', 'shining', 'crawling'])} things")

        # Create a myth 
        myth_subject = random.choice(["the first being", "the great flood", "the sky fire", "the wandering star"])
        myth_lesson = random.choice(["obey the spirits", "share food", "fear the night", "honor ancestors"])
        self.belief.myths.append(f"The myth of {myth_subject} teaches us to {myth_lesson}")

    def get_current_biome(self):
        x_idx = min(max(0, int(self.x // 10)), len(elevation) - 1)
        y_idx = min(max(0, int(self.y // 10)), len(elevation[0]) - 1)
        elev = elevation[x_idx][y_idx]
        temp = temperature[x_idx][y_idx]
        moist = moisture[x_idx][y_idx]
        if elev < -0.2:
            return Biome.OCEAN
        elif elev < -0.1:
            return Biome.COAST
        elif elev > 0.4:
            return Biome.MOUNTAIN
        elif temp < -0.3:
            return Biome.TUNDRA
        elif moist < -0.3:
            return Biome.DESERT
        elif moist > 0.3 and temp > 0:
            return Biome.FOREST
        elif abs(moist) < 0.2 and temp > 0:
            return Biome.PLAINS
        return Biome.PLAINS

    def biome_happiness(self):
        current_biome = self.get_current_biome()
        return 1.0 if current_biome == self.dna.preferred_biome else 0.7

    def update(self, organisms, foods, tribes, settlements, current_time):
        if self.dreaming:
            self.dream()
            return
        self.age += 1
        weather_effect = 1.0 
        if weather_state == "rain": 
            weather_effect = 0.8
        elif weather_state == "storm": 
            weather_effect = 0.6
        self.energy -= self.dna.metabolism * (1.0 + (0.5 if self.disease else 0)) * weather_effect # Modified for weather effect
        self.apply_disease_effects(current_time)
        if self.age > self.max_age or self.energy <= 0:
            self.die(organisms, tribes)
            return
        self.update_emotions()
        if current_phase.value >= Phase.DREAM.value and not self.dreaming:
            if random.random() < 0.002 * self.dna.dream_recall:
                self.start_dreaming()
        if self.emotion.hunger > 0.6:
            self.find_food(foods)
        elif self.home and random.random() < 0.05 and current_phase.value >= Phase.CIVILIZATION.value:
            self.move_toward(self.home.x, self.home.y)
        if self.role == Role.GATHERER and self.energy > 50 and len(foods) > 0:
            self.perform_gathering(foods)
        elif self.role == Role.WARRIOR and self.tribe and self.tribe.enemies:
            self.perform_warrior_duties(organisms, tribes)
        elif self.role == Role.BUILDER and self.home and self.home.food_storage > 50:
            self.perform_building(settlements)
        if not self.path and random.random() < 0.1:
            self.move()
        if self.path:
            self.follow_path()
        self.eat(foods)
        if current_phase.value >= Phase.EMOTION.value:
            self.social_interactions(organisms, tribes)
        if (self.energy > 100 and self.age > 100 and # Easier reproduction
            current_time - self.last_reproduction_time > 100 / simulation_speed): # Earlier reproduction
            self.reproduce(organisms, tribes)
        self.rect = pygame.Rect(self.x - self.dna.size, self.y - self.dna.size, self.dna.size * 2, self.dna.size * 2) 

    def apply_disease_effects(self, current_time):
        if not self.disease:
            return

        # Apply symptoms
        if "weakness" in self.disease.symptoms:
            self.dna.speed *= 0.7
        if "rage" in self.disease.symptoms and random.random() < 0.05:
            self.emotion.rage = min(1.0, self.emotion.rage + 0.3)
        if "paralysis" in self.disease.symptoms and random.random() < 0.01:
            self.path = []  # Stop moving

        # Chance of death 
        if random.random() < self.disease.mortality_rate / 1000:
            self.energy = 0

        # Chance to recover 
        if current_time - self.disease_start_time > self.disease.duration:
            if random.random() < self.dna.immunity:
                # Develop immunity
                self.immunities[self.disease.disease_id] = random.uniform(0.7, 1.0)
                self.disease = None
                self.symptoms = []
            else:
                # Disease mutates
                self.disease.mutate()
                self.disease_start_time = current_time

    def update_emotions(self):
        # Increase hunger over time 
        self.emotion.hunger = min(1.0, self.emotion.hunger + 0.001)

        # Fear based on health and environment 
        self.emotion.fear = (1 - (self.energy / 100)) * 0.5
        if self.get_current_biome() != self.dna.preferred_biome:
            self.emotion.fear = min(1.0, self.emotion.fear + 0.1)

        # Happiness based on biome match 
        self.emotion.joy = self.biome_happiness() * 0.5

        # Random emotional fluctuations 
        self.emotion.rage = max(0, self.emotion.rage + random.uniform(-0.02, 0.02))
        self.emotion.joy = max(0, self.emotion.joy + random.uniform(-0.02, 0.02))
        self.emotion.loyalty = min(1.0, self.emotion.loyalty + random.uniform(-0.01, 0.01))
        if weather_state == "storm": 
            self.emotion.fear = min(1.0, self.emotion.fear + 0.2)
            self.emotion.curiosity = max(0, self.emotion.curiosity - 0.1)

    def move(self):
        # Simple random movement with biome preference
        if self.dna.preferred_biome != self.get_current_biome() and random.random() < 0.7:
            # Try to move toward preferred biome
            angle = random.uniform(0, 2 * math.pi)
            test_x = self.x + math.cos(angle) * 100
            test_y = self.y + math.sin(angle) * 100
            if 0 <= test_x < WIDTH and 0 <= test_y < HEIGHT:
                test_biome = self.get_current_biome_at(test_x, test_y)
                if test_biome == self.dna.preferred_biome:
                    self.x += math.cos(angle) * self.dna.speed
                    self.y += math.sin(angle) * self.dna.speed
                    return

        # Otherwise random move
        angle = random.uniform(0, 2 * math.pi)
        speed = self.dna.speed * (1 - self.emotion.fear * 0.5)
        self.x += math.cos(angle) * speed
        self.y += math.sin(angle) * speed

        # Keep within bounds
        self.x = max(10, min(WIDTH - 10, self.x))
        self.y = max(10, min(HEIGHT - 10, self.y))

    def get_current_biome_at(self, x, y):
        x_idx = min(max(0, int(x // 10)), len(elevation) - 1)
        y_idx = min(max(0, int(y // 10)), len(elevation[0]) - 1)

        elev = elevation[x_idx][y_idx]
        temp = temperature[x_idx][y_idx]
        moist = moisture[x_idx][y_idx]

        if elev < -0.2:
            return Biome.OCEAN
        elif elev < -0.1:
            return Biome.COAST
        elif elev > 0.4:
            return Biome.MOUNTAIN
        elif temp < -0.3:
            return Biome.TUNDRA
        elif moist < -0.3:
            return Biome.DESERT
        elif moist > 0.3 and temp > 0:
            return Biome.FOREST
        elif abs(moist) < 0.2 and temp > 0:
            return Biome.PLAINS
        return Biome.PLAINS

    def move_toward(self, target_x, target_y):
        angle = math.atan2(target_y - self.y, target_x - self.x)
        speed = self.dna.speed * (1 - self.emotion.fear * 0.5)
        self.x += math.cos(angle) * speed
        self.y += math.sin(angle) * speed

        # Keep within bounds
        self.x = max(10, min(WIDTH - 10, self.x))
        self.y = max(10, min(HEIGHT - 10, self.y))

    def follow_path(self):
        if not self.path:
            return

        target_x, target_y = self.path[0]
        dist = math.sqrt((self.x - target_x)**2 + (self.y - target_y)**2)

        if dist < 5:
            self.path.pop(0)
        else:
            self.move_toward(target_x, target_y)

    def find_food(self, foods):
        if not foods:
            return

        # Find closest food
        closest = None
        min_dist = float('inf')

        for food in foods:
            dist = math.sqrt((self.x - food.x)**2 + (self.y - food.y)**2)
            if dist < min_dist:
                min_dist = dist
                closest = food

        # Move toward food
        if closest and min_dist < self.dna.vision_range:
            self.move_toward(closest.x, closest.y)

    def perform_gathering(self, foods):
        if not foods:
            return

        # Find closest food
        closest = None
        min_dist = float('inf')

        for food in foods:
            dist = math.sqrt((self.x - food.x)**2 + (self.y - food.y)**2)
            if dist < min_dist:
                min_dist = dist
                closest = food

        if closest and min_dist < self.dna.vision_range:
            if min_dist > 10:
                self.move_toward(closest.x, closest.y)
            elif self.home:
                # Bring food to home settlement
                if min_dist < 10:
                    self.home.food_storage += closest.energy
                    foods.remove(closest)
                    self.emotion.joy = min(1.0, self.emotion.joy + 0.05)
                elif not self.path:
                    self.path = self.create_path(self.home.x, self.home.y)

    def perform_warrior_duties(self, organisms, tribes):
        if not self.tribe or not self.tribe.enemies:
            return

        # Find closest enemy
        closest_enemy = None
        min_dist = float('inf')

        for org in organisms:
            if org.tribe in self.tribe.enemies:
                dist = math.sqrt((self.x - org.x)**2 + (self.y - org.y)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_enemy = org

        if closest_enemy:
            if min_dist < 20:
                # Attack
                damage = self.dna.aggression * 10
                closest_enemy.energy -= damage
                self.emotion.rage = min(1.0, self.emotion.rage + 0.05)

                if closest_enemy.energy <= 0:
                    # Log historical event
                    event = HistoricalEvent(
                        time=time.time(),
                        event_type=EventType.WAR,
                        description=f"{self.tribe.tribe_id} warrior killed {closest_enemy.id} from {closest_enemy.tribe.tribe_id}",
                        location=(self.x, self.y),
                        involved=[self, closest_enemy]
                    )
                    heapq.heappush(historical_events, event)
            elif min_dist < self.dna.vision_range:
                self.move_toward(closest_enemy.x, closest_enemy.y)

    def perform_building(self, settlements):
        if not self.home or not self.home.food_storage > 50:
            return

        # Find a random spot near settlement to build
        build_x = self.home.x + random.uniform(-100, 100)
        build_y = self.home.y + random.uniform(-100, 100)

        if not self.path:
            self.path = self.create_path(build_x, build_y)

        if math.sqrt((self.x - build_x)**2 + (self.y - build_y)**2) < 10:
            # Build a structure
            structure_type = random.choice([
                StructureType.HUT,
                StructureType.STORAGE,
                StructureType.TEMPLE if random.random() < 0.3 else StructureType.HUT
            ])

            new_structure = Structure(
                structure_id=uuid.uuid4().hex[:6],
                structure_type=structure_type,
                x=build_x,
                y=build_y,
                size=random.uniform(15, 30),
                owner=self
            )

            self.home.structures.append(new_structure)
            self.home.food_storage -= 50
            self.emotion.joy = min(1.0, self.emotion.joy + 0.1)

            # Log historical event
            event = HistoricalEvent(
                time=time.time(),
                event_type=EventType.SETTLEMENT_BUILT,
                description=f"New {structure_type.name.lower()} built in {self.home.settlement_id}",
                location=(build_x, build_y),
                involved=[self]
            )
            heapq.heappush(historical_events, event)

    def create_path(self, target_x, target_y):
        # Simple pathfinding - straight line with intermediate points
        path = []
        steps = max(2, int(math.sqrt((target_x - self.x)**2 + (target_y - self.y)**2) / 50))

        for i in range(1, steps + 1):
            fraction = i / steps
            path.append((
                self.x + (target_x - self.x) * fraction,
                self.y + (target_y - self.y) * fraction
            ))

        return path

    def eat(self, foods):
        for food in foods[:]:
            dist = math.sqrt((self.x - food.x)**2 + (self.y - food.y)**2)
            if dist < self.dna.size + 5:
                energy_gain = food.energy * (1.2 if self.get_current_biome() == self.dna.preferred_biome else 1.0)
                self.energy = min(200, self.energy + energy_gain)
                foods.remove(food)
                self.emotion.hunger = max(0, self.emotion.hunger - 0.4)
                self.emotion.joy = min(1.0, self.emotion.joy + 0.15)
                self.memory.events.append(("eat", self.age, food.x, food.y))
                break

    def social_interactions(self, organisms, tribes):
        # Find nearby organisms
        nearby = []
        for org in organisms:
            if org != self:
                dist = math.sqrt((self.x - org.x)**2 + (self.y - org.y)**2)
                if dist < self.dna.vision_range:
                    nearby.append((org, dist))

        if not nearby:
            return

        # Sort by distance
        nearby.sort(key=lambda x: x[1])
        closest_org, dist = nearby[0]

        # Fear response
        if (closest_org.dna.size > self.dna.size * 1.5 and self.emotion.fear > 0.3 and
            (not self.tribe or closest_org.tribe != self.tribe)):
            # Move away from larger organisms
            angle = math.atan2(self.y - closest_org.y, self.x - closest_org.x)
            self.x += math.cos(angle) * self.dna.speed * 0.8
            self.y += math.sin(angle) * self.dna.speed * 0.8
            self.memory.events.append(("fear", self.age, closest_org.id))

        # Social bonding
        elif (self.dna.sociability > 0.5 and dist < 30 and self.emotion.fear < 0.4 and
              closest_org.tribe == self.tribe):
            self.emotion.joy = min(1.0, self.emotion.joy + 0.02)
            closest_org.emotion.joy = min(1.0, closest_org.emotion.joy + 0.02)
            self.memory.events.append(("bond", self.age, closest_org.id))

            # Share beliefs and language
            if current_phase.value >= Phase.LANGUAGE.value:
                if random.random() < 0.1:
                    self.share_beliefs(closest_org)
                if random.random() < 0.1:
                    self.share_language(closest_org)

        # Tribe formation
        if (current_phase.value >= Phase.TRIBAL.value and not self.tribe and
            dist < 50 and self.dna.sociability > 0.6 and closest_org.dna.sociability > 0.6):
            self.form_tribe(closest_org, tribes)

    def share_beliefs(self, other):
        # Share a random belief
        if hasattr(self, 'belief'): # Added hasattr check
            if self.belief.origin_story and random.random() < 0.5:
                other.belief.origin_story = self.belief.origin_story
            elif self.belief.deities and random.random() < 0.5:
                deity = random.choice(self.belief.deities)
                if deity not in other.belief.deities:
                    other.belief.deities.append(deity)
            elif self.belief.myths and random.random() < 0.5:
                myth = random.choice(self.belief.myths)
                if myth not in other.belief.myths:
                    other.belief.myths.append(myth)

    def share_language(self, other):
        # Share a random glyph meaning
        if self.language and other.language:
            self_glyph, self_meaning = random.choice(list(self.language.items()))
            other.language[self_glyph] = self_meaning

    def form_tribe(self, other, tribes):
        # Create a new tribe 
        new_tribe = Tribe(
            tribe_id=f"Tribe-{uuid.uuid4().hex[:4]}",
            color=(random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        )

        new_tribe.members.append(self)
        new_tribe.members.append(other)
        self.tribe = new_tribe
        other.tribe = new_tribe

        # Create shared beliefs 
        if hasattr(self, 'belief') and self.belief.origin_story: # Added hasattr check
            new_tribe.shared_beliefs.origin_story = self.belief.origin_story
        elif hasattr(other, 'belief') and other.belief.origin_story: # Added hasattr check
            new_tribe.shared_beliefs.origin_story = other.belief.origin_story

        # Create shared language 
        for glyph, meaning in self.language.items():
            new_tribe.language[glyph] = meaning
        for glyph, meaning in other.language.items():
            if glyph not in new_tribe.language:
                new_tribe.language[glyph] = meaning

        tribes.append(new_tribe)

        # Log historical event 
        event = HistoricalEvent(
            time=time.time(),
            event_type=EventType.TRIBE_FORMED,
            description=f"New tribe formed: {new_tribe.tribe_id}",
            location=((self.x + other.x)/2, (self.y + other.y)/2),
            involved=[self, other]
        )
        heapq.heappush(historical_events, event)

    def reproduce(self, organisms, tribes):
        if self.energy < 100: # Easier reproduction
            return

        # Find mate  
        mate = None
        for org in organisms:
            if (org != self and org.energy > 90 and # Easier reproduction
                math.sqrt((self.x - org.x)**2 + (self.y - org.y)**2) < 50 and
                (not self.tribe or org.tribe == self.tribe)):
                mate = org
                break

        if mate:
            # Create offspring 
            self.energy -= 50 # Reduced energy cost
            mate.energy -= 50 # Reduced energy cost
            offspring = Organism(
                (self.x + mate.x)/2, (self.y + mate.y)/2, parents=[self, mate]
            )
            # Inherit tribe
            if self.tribe:
                offspring.tribe = self.tribe
                self.tribe.members.append(offspring)
            organisms.append(offspring)
            self.last_reproduction_time = time.time()
            self.memory.events.append(("reproduce", self.age, mate.id))

            # Log historical event
            event = HistoricalEvent(
                time=time.time(),
                event_type=EventType.REPRODUCTION,
                description=f"New offspring born to {self.id} and {mate.id}",
                location=((self.x + mate.x)/2, (self.y + mate.y)/2),
                involved=[self, mate, offspring]
            )
            heapq.heappush(historical_events, event)
            # print(f"Mating Event: Parent1={self.id} (Size={self.dna.size:.2f}, Shape={self.dna.shape}), " 
            #       f"Parent2={mate.id} (Size={mate.dna.size:.2f}, Shape={mate.dna.shape}), "
            #       f"Offspring={offspring.id} (Size={offspring.dna.size:.2f}, Shape={offspring.dna.shape}) "
            #       f"at ({offspring.x:.1f}, {offspring.y:.1f})")

    def start_dreaming(self):
        self.dreaming = True
        self.dream_start_time = time.time()
        self.dream_duration = random.uniform(2.0, 5.0)
        self.memory.events.append(("dream_start", self.age))

    def dream(self):
        # Dreaming logic 
        current_time = time.time()
        if current_time - self.dream_start_time > self.dream_duration:
            self.dreaming = False
            self.memory.events.append(("dream_end", self.age))
            self.form_beliefs_from_dream()
            return

        # Random dream content based on memories 
        if self.memory.events:
            memory = random.choice(list(self.memory.events))
            self.memory.dreams.append(memory)

    def form_beliefs_from_dream(self):
        if not self.memory.dreams:
            return
        dream = random.choice(self.memory.dreams)
        event_type = dream[0]
        if event_type == "trauma":
            # Form protective belief 
            belief = f"The {random.choice(['Spirits', 'Ancients', 'Dreamwalkers', 'Star-Beings'])} warn of {dream[2]}"
            self.belief.rituals.append(belief)
            # Share with tribe 
            if self.tribe and random.random() < 0.3:
                self.tribe.shared_beliefs.rituals.append(belief)
        elif event_type == "bond":
            # Form social belief 
            belief = f"Bonding brings favor from the {random.choice(['Sky', 'Earth', 'Water', 'Moon'])} Spirits"
            self.belief.rituals.append(belief)
        elif event_type == "death":
            # Form afterlife belief 
            belief = f"After death, we join the {random.choice(['Eternal Light', 'Great Dream', 'Silent Void', 'Star Council'])}"
            self.belief.deities.append(belief)
            # Create a myth 
            myth = f"How {random.choice(['the First Being', 'the Sky God', 'the Earth Mother'])} conquered death"
            self.belief.myths.append(myth)
            # Log historical event 
            event = HistoricalEvent(
                time=time.time(),
                event_type=EventType.MYTH_CREATED,
                description=f"New myth created: {myth}",
                location=(self.x, self.y),
                involved=[self]
            )
            heapq.heappush(historical_events, event)

    def die(self, organisms, tribes):
        # Create corpse 
        global corpses
        corpses.append((self.x, self.y, self.dna.size))
        # Remove from organisms list 
        if self in organisms:
            organisms.remove(self)
        # Remove from tribe 
        if self.tribe and self in self.tribe.members:
            self.tribe.members.remove(self)
        if self.role == Role.LEADER and self.tribe and self.tribe.members: # Added check for tribe.members
            # Appoint new leader 
            new_leader = max(self.tribe.members, key=lambda m: m.dna.intelligence)
            new_leader.role = Role.LEADER
            self.tribe.leader = new_leader
        # Add to memory of nearby organisms 
        for org in organisms:
            dist = math.sqrt((self.x - org.x)**2 + (self.y - org.y)**2)
            if dist < org.dna.vision_range:
                org.memory.events.append(("death", org.age, self.id))
                org.emotion.sorrow = min(1.0, org.emotion.sorrow + 0.3)
        # print(f"Organism {self.id} died at age {self.age}") 

    def draw(self, screen_surface):
        # Calculate emotional color safely 
        r = self.dna.color[0] * (1 - self.emotion.fear) + 120 * self.emotion.rage
        g = self.dna.color[1] * (1 - self.emotion.rage) + 180 * self.emotion.joy
        b = self.dna.color[2] * (1 - self.emotion.hunger) + 220 * self.emotion.fear
        r = int(max(0, min(255, r * brightness))) # Apply brightness
        g = int(max(0, min(255, g * brightness))) # Apply brightness
        b = int(max(0, min(255, b * brightness))) # Apply brightness
        color = (r, g, b)

        # Apply camera and zoom transformation 
        display_x = self.x * zoom + camera_x
        display_y = self.y * zoom + camera_y
        display_size = self.dna.size * zoom

        # Add base glow for visibility 
        glow_size = display_size + 6
        glow_surf = pygame.Surface((glow_size*2, glow_size*2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*color, 40), (glow_size, glow_size), glow_size)
        screen_surface.blit(glow_surf, (display_x - glow_size, display_y - glow_size))

        # Dream aura or disease aura 
        if self.dreaming:
            aura_size = display_size + 10
            aura_surf = pygame.Surface((aura_size*2, aura_size*2), pygame.SRCALPHA)
            pygame.draw.circle(aura_surf, (100, 200, 255, 80), (aura_size, aura_size), aura_size)
            screen_surface.blit(aura_surf, (display_x - aura_size, display_y - aura_size))
        elif self.disease and show_disease:
            aura_size = display_size + 12
            aura_surf = pygame.Surface((aura_size*2, aura_size*2), pygame.SRCALPHA)
            disease_color_bright = tuple(int(c * brightness) for c in self.disease.color)
            pygame.draw.circle(aura_surf, (*disease_color_bright, 70), (aura_size, aura_size), aura_size)
            screen_surface.blit(aura_surf, (display_x - aura_size, display_y - aura_size))

        # Draw organism shape 
        if self.dna.shape == 'circle':
            pygame.draw.circle(screen_surface, color, (int(display_x), int(display_y)), int(display_size))
        elif self.dna.shape == 'triangle': 
            points = [
                (display_x, display_y - display_size),
                (display_x - display_size * 0.866, display_y + display_size * 0.5),
                (display_x + display_size * 0.866, display_y + display_size * 0.5)
            ]
            pygame.draw.polygon(screen_surface, color, points)
        elif self.dna.shape == 'square': 
            pygame.draw.rect(screen_surface, color, (display_x - display_size, display_y - display_size, display_size * 2, display_size * 2))
        else:  # pentagon 
            points = []
            for i in range(5):
                angle = math.radians(i * 72 - 90)
                px = display_x + display_size * math.cos(angle)
                py = display_y + display_size * math.sin(angle)
                points.append((px, py))
            pygame.draw.polygon(screen_surface, color, points)

        # Draw limbs 
        for i in range(self.dna.limb_count):
            angle = math.radians(i * (360 / self.dna.limb_count))
            limb_length = display_size * 1.5
            end_x = display_x + limb_length * math.cos(angle)
            end_y = display_y + limb_length * math.sin(angle)
            pygame.draw.line(screen_surface, color, (int(display_x), int(display_y)), (int(end_x), int(end_y)), max(1, int(3 * zoom)))

        # Draw glyph if language is enabled 
        if current_phase.value >= Phase.LANGUAGE.value and show_language:
            # Draw black outline 
            outline_surf = font_medium.render(self.glyph, True, (0, 0, 0))
            screen_surface.blit(outline_surf, (display_x - outline_surf.get_width()/2 + 1, display_y - display_size - 15 + 1))
            glyph_surf = font_medium.render(self.glyph, True, (255, 255, 255)) # White text 
            screen_surface.blit(glyph_surf, (display_x - glyph_surf.get_width()/2, display_y - display_size - 15))

        # Draw role icon (Unicode symbols) 
        if self.role and current_phase.value >= Phase.CIVILIZATION.value:
            role_icon = ""
            if self.role == Role.LEADER:
                role_icon = u"\u2691"  # ⚑
            elif self.role == Role.PRIEST:
                role_icon = u"\u2600"  # ☀
            elif self.role == Role.WARRIOR:
                role_icon = u"\u25B2"  # ▲
            elif self.role == Role.GATHERER:
                role_icon = u"\u2665"  # ♥
            elif self.role == Role.BUILDER:
                role_icon = u"\u2692"  # ⚒
            elif self.role == Role.HEALER:
                role_icon = u"\u271A"  # ✚

            # Draw black outline 
            outline_surf = font_medium.render(role_icon, True, (0, 0, 0))
            screen_surface.blit(outline_surf, (display_x - outline_surf.get_width()/2 + 1, display_y - display_size - 30 + 1))
            role_surf = font_medium.render(role_icon, True, (255, 255, 255)) # White text 
            screen_surface.blit(role_surf, (display_x - role_surf.get_width()/2, display_y - display_size - 30))

        # Optional: show path 
        if show_paths and self.path:
            # Transform path coordinates 
            transformed_path = [(p_x * zoom + camera_x, p_y * zoom + camera_y) for p_x, p_y in self.path]
            for i in range(len(transformed_path) - 1):
                pygame.draw.line(screen_surface, (200, 150, 100, 150), transformed_path[i], transformed_path[i+1], max(1, int(2 * zoom)))

    def to_dict(self): 
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "energy": self.energy,
            "age": self.age,
            "max_age": self.max_age,
            "dna": {
                "color": self.dna.color,
                "size": self.dna.size,
                "speed": self.dna.speed,
                "aggression": self.dna.aggression,
                "sociability": self.dna.sociability,
                "intelligence": self.dna.intelligence,
                "dream_recall": self.dna.dream_recall,
                "vision_range": self.dna.vision_range,
                "metabolism": self.dna.metabolism,
                "limb_count": self.dna.limb_count,
                "shape": self.dna.shape,
                "immunity": self.dna.immunity,
                "mutation_rate": self.dna.mutation_rate,
                "preferred_biome": self.dna.preferred_biome.name if self.dna.preferred_biome else None
            },
            "emotion": vars(self.emotion),
            "role": self.role.name if self.role else None,
            "tribe_id": self.tribe.tribe_id if self.tribe else None,
            "home_id": self.home.settlement_id if self.home else None,
            "disease_id": self.disease.disease_id if self.disease else None,
            # For simplicity, Memory and Belief objects are not fully serialized here,
            # as they would require deeper serialization logic.
            # Only basic attributes are included for now.
        }

    @staticmethod
    def from_dict(data): 
        dna = DNA()
        dna.color = tuple(data["dna"]["color"])
        dna.size = data["dna"]["size"]
        dna.speed = data["dna"]["speed"]
        dna.aggression = data["dna"]["aggression"]
        dna.sociability = data["dna"]["sociability"]
        dna.intelligence = data["dna"]["intelligence"]
        dna.dream_recall = data["dna"]["dream_recall"]
        dna.vision_range = data["dna"]["vision_range"]
        dna.metabolism = data["dna"]["metabolism"]
        dna.limb_count = data["dna"]["limb_count"]
        dna.shape = data["dna"]["shape"]
        dna.immunity = data["dna"]["immunity"]
        dna.mutation_rate = data["dna"]["mutation_rate"]
        dna.preferred_biome = Biome[data["dna"]["preferred_biome"]] if data["dna"]["preferred_biome"] else None

        emotion = Emotion()
        emotion.__dict__.update(data["emotion"])

        org = Organism(x=data["x"], y=data["y"], dna=dna)
        org.id = data["id"]
        org.energy = data["energy"]
        org.age = data["age"]
        org.max_age = data["max_age"]
        org.emotion = emotion
        org.role = Role[data["role"]] if data["role"] else None
        # Tribe, Home (Settlement), and Disease will be re-linked in World.load_simulation
        org.temp_tribe_id = data.get("tribe_id")
        org.temp_home_id = data.get("home_id")
        org.temp_disease_id = data.get("disease_id")
        return org

    def get_stats(self): 
        return (f"ID: {self.id} | Age: {self.age} | Energy: {self.energy:.1f} | "
                f"Emotion: Fear={self.emotion.fear:.2f}, Rage={self.emotion.rage:.2f}, "
                f"Hunger={self.emotion.hunger:.2f}, Joy={self.emotion.joy:.2f}, "
                f"Curiosity={self.emotion.curiosity:.2f}, Loyalty={self.emotion.loyalty:.2f} | "
                f"Genes: Size={self.dna.size:.2f}, Shape={self.dna.shape}, Speed={self.dna.speed:.2f}, "
                f"Aggression={self.dna.aggression:.2f}, Sociability={self.dna.sociability:.2f}, "
                f"Intelligence={self.dna.intelligence:.2f}, DreamRecall={self.dna.dream_recall:.2f} | "
                f"Tribe: {self.tribe.tribe_id if self.tribe else 'None'} | "
                f"Role: {self.role.name if self.role else 'None'} | "
                f"Biome: {self.get_current_biome().name}")


class Food:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.energy = random.uniform(10, 30)
        self.size = 5
        self.biome = self.get_biome_at(x, y) 

    def get_biome_at(self, x, y): 
        x_idx = min(max(0, int(x // 10)), len(elevation) - 1)
        y_idx = min(max(0, int(y // 10)), len(elevation[0]) - 1)
        elev = elevation[x_idx][y_idx]
        temp = temperature[x_idx][y_idx]
        moist = moisture[x_idx][y_idx]
        if elev < -0.2:
            return Biome.OCEAN
        elif elev < -0.1:
            return Biome.COAST
        elif elev > 0.4:
            return Biome.MOUNTAIN
        elif temp < -0.3:
            return Biome.TUNDRA
        elif moist < -0.3:
            return Biome.DESERT
        elif moist > 0.3 and temp > 0:
            return Biome.FOREST
        elif abs(moist) < 0.2 and temp > 0:
            return Biome.PLAINS
        return Biome.PLAINS

    def draw(self, screen_surface):
        # Apply camera and zoom transformation 
        display_x = self.x * zoom + camera_x
        display_y = self.y * zoom + camera_y
        display_size = self.size * zoom

        # Apply brightness to food color 
        food_color_bright = tuple(int(c * brightness) for c in FOOD_COLOR)

        pygame.draw.circle(screen_surface, food_color_bright, (int(display_x), int(display_y)), int(display_size))


# Global list for corpses (simple, not part of Organism lifecycle)
corpses = []

class World:
    def __init__(self):
        self.organisms = []
        self.foods = []
        self.tribes = []
        self.settlements = []
        self.corpses = []
        self.current_time = 0
        self.event_log = deque(maxlen=20) # Keep last 20 events for display 
        self.diseases = {} # Store active diseases 

        self.spawn_initial_organisms(50)
        self.spawn_initial_food(100)

    def spawn_initial_organisms(self, count):
        for _ in range(count):
            x = random.randint(50, WIDTH - 50)
            y = random.randint(50, HEIGHT - 50)
            org = Organism(x, y) 
            self.organisms.append(org)
            # print(f"Organism {org.id} born at ({x}, {y})") 

    def spawn_initial_food(self, count):
        for _ in range(count):
            x = random.randint(50, WIDTH - 50)
            y = random.randint(50, HEIGHT - 50)
            self.foods.append(Food(x, y))

    def update(self):
        global current_phase
        self.current_time += 1 * simulation_speed

        # Advance evolution phase 
        self.advance_phase()

        # Update organisms 
        for organism in self.organisms[:]:
            organism.update(self.organisms, self.foods, self.tribes, self.settlements, self.current_time)

        # Update tribes 
        for tribe in self.tribes[:]:
            self.update_tribe(tribe)
            if not tribe.members:
                self.tribes.remove(tribe)

        # Update settlements 
        for settlement in self.settlements[:]:
            self.update_settlement(settlement)
            if not settlement.population:
                self.settlements.remove(settlement)

        # Replenish food  (modified for consistent food regeneration)
        food_spawn_rate = 0.5 # Increased food spawn rate
        if weather_state == "rain": 
            food_spawn_rate *= 1.5
        elif weather_state == "storm": 
            food_spawn_rate *= 0.5

        if random.random() < food_spawn_rate:
            x = random.randint(50, WIDTH - 50)
            y = random.randint(50, HEIGHT - 50)
            self.foods.append(Food(x, y))

        # Clean up corpses 
        self.corpses = [(x, y, s) for x, y, s in self.corpses if random.random() > 0.005] # Slower decay

        # Process historical events 
        while historical_events:
            event = heapq.heappop(historical_events)
            self.event_log.append(f"Day {int(self.current_time/100)}: {event.description}")
            if len(self.event_log) > 20:
                self.event_log.popleft()

        # Disease spread and mutation 
        self.manage_diseases()

    def advance_phase(self):
        global current_phase
        num_organisms = len(self.organisms)
        num_tribes = len(self.tribes)
        avg_intelligence = np.mean([o.dna.intelligence for o in self.organisms]) if self.organisms else 0
        avg_sociability = np.mean([o.dna.sociability for o in self.organisms]) if self.organisms else 0

        new_phase = current_phase

        if current_phase == Phase.BLOB and num_organisms > 200:
            new_phase = Phase.HUNTER_GATHERER
        elif current_phase == Phase.HUNTER_GATHERER and avg_intelligence > 0.3 and self.current_time > 5000:
            new_phase = Phase.EMOTION
        elif current_phase == Phase.EMOTION and avg_intelligence > 0.4 and self.current_time > 10000:
            new_phase = Phase.DREAM
        elif current_phase == Phase.DREAM and self.current_time > 15000 and random.random() < 0.0001:
            # Cannibalism phase - rare trigger 
            new_phase = Phase.CANNIBAL
        elif current_phase == Phase.CANNIBAL and self.current_time > 20000 and num_tribes > 0:
            new_phase = Phase.TRIBAL
        elif current_phase == Phase.TRIBAL and avg_intelligence > 0.5 and num_tribes > 5 and self.current_time > 25000:
            new_phase = Phase.LANGUAGE
            self.introduce_language()
        elif current_phase == Phase.LANGUAGE and avg_sociability > 0.6 and num_tribes > 10 and self.current_time > 30000:
            new_phase = Phase.CONFLICT
        elif current_phase == Phase.CONFLICT and random.random() < 0.00005 and self.current_time > 35000:
            new_phase = Phase.DISEASE
            self.introduce_disease()
        elif current_phase == Phase.DISEASE and num_tribes > 15 and self.current_time > 40000:
            new_phase = Phase.CIVILIZATION
            self.form_settlements()

        if new_phase != current_phase:
            # print(f"Advancing to new phase: {new_phase.name}") 
            event = HistoricalEvent( 
                time=self.current_time,
                event_type=EventType.DISCOVERY,
                description=f"New phase: {new_phase.name}",
                location=(WIDTH//2, HEIGHT//2)
            )
            heapq.heappush(historical_events, event)
            current_phase = new_phase

    def introduce_language(self):
        # Assign basic glyph meanings 
        for org in self.organisms:
            if not org.language:
                org.language["△"] = "food"
                org.language["◯"] = "danger"
                org.language["□"] = "water"
                org.language["★"] = "dream"
                org.language["☮"] = "peace"
                org.language["⚔"] = "war"

        for tribe in self.tribes:
            if not tribe.language:
                tribe.language["△"] = "food"
                tribe.language["◯"] = "danger"
                tribe.language["□"] = "water"

    def introduce_disease(self):
        
        disease_types = list(DiseaseType)
        new_disease = Disease(
            disease_id=f"Disease-{uuid.uuid4().hex[:4]}",
            disease_type=random.choice(disease_types),
            transmission_rate=random.uniform(0.01, 0.1),
            mortality_rate=random.uniform(0.001, 0.02),
            mutation_rate=random.uniform(0.0001, 0.001),
            immunity_resistance=random.uniform(0.0, 0.3),
            symptoms=["weakness"],
            color=(random.randint(100, 200), random.randint(0, 50), random.randint(0, 50))
        )
        self.diseases[new_disease.disease_id] = new_disease # Store disease globally
        # Infect a small number of organisms
        for _ in range(max(1, len(self.organisms) // 20)):
            organism = random.choice(self.organisms)
            organism.disease = new_disease
            organism.disease_start_time = self.current_time

        event = HistoricalEvent(
            time=self.current_time,
            event_type=EventType.DISEASE_OUTBREAK,
            description=f"A new disease ({new_disease.disease_type.name}) outbreak!",
            location=(WIDTH/2, HEIGHT/2)
        )
        heapq.heappush(historical_events, event)

    def manage_diseases(self):
        for organism in self.organisms[:]:
            if organism.disease:
                # Spread disease 
                for other_org in self.organisms:
                    if other_org != organism and not other_org.disease:
                        dist = math.sqrt((organism.x - other_org.x)**2 + (organism.y - other_org.y)**2)
                        if dist < 30 and random.random() < organism.disease.transmission_rate:
                            # Check immunity 
                            if random.random() > other_org.dna.immunity + other_org.immunities.get(organism.disease.disease_id, 0):
                                other_org.disease = organism.disease
                                other_org.disease_start_time = self.current_time
                                other_org.symptoms.extend(organism.disease.symptoms)

    def form_settlements(self):
        # Create initial settlements from existing tribes 
        for tribe in self.tribes:
            if not tribe.settlements and len(tribe.members) > 5:
                leader = max(tribe.members, key=lambda m: m.dna.intelligence)
                settlement_x = leader.x + random.uniform(-50, 50)
                settlement_y = leader.y + random.uniform(-50, 50)

                new_settlement = Settlement(
                    settlement_id=f"Settlement-{uuid.uuid4().hex[:4]}",
                    x=settlement_x,
                    y=settlement_y,
                    radius=50,
                    tribe=tribe
                )
                tribe.settlements.append(new_settlement)
                self.settlements.append(new_settlement)

                # Assign roles to tribe members and set home 
                for member in tribe.members:
                    member.home = new_settlement
                    if member == leader:
                        member.role = Role.LEADER
                    elif member.dna.sociability > 0.7:
                        member.role = random.choice([Role.PRIEST, Role.HEALER])
                    elif member.dna.aggression > 0.7:
                        member.role = Role.WARRIOR
                    elif member.dna.intelligence > 0.6:
                        member.role = Role.BUILDER
                    else:
                        member.role = Role.GATHERER

                event = HistoricalEvent(
                    time=self.current_time,
                    event_type=EventType.SETTLEMENT_BUILT,
                    description=f"{tribe.tribe_id} established settlement {new_settlement.settlement_id}",
                    location=(settlement_x, settlement_y),
                    involved=[leader]
                )
                heapq.heappush(historical_events, event)

    def update_tribe(self, tribe):
        # Update tribe's territory based on members 
        if tribe.members:
            all_x = [m.x for m in tribe.members]
            all_y = [m.y for m in tribe.members]
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
            tribe.territory = [(min_x - 20, min_y - 20), (max_x + 20, max_y + 20)]

        # Check for leadership 
        if not tribe.leader and tribe.members:
            tribe.leader = max(tribe.members, key=lambda m: m.dna.intelligence)
            tribe.leader.role = Role.LEADER

        # Conflict with other tribes 
        if current_phase.value >= Phase.CONFLICT.value and self.current_time - tribe.last_war_time > 1000:
            for other_tribe in self.tribes:
                if other_tribe != tribe and other_tribe not in tribe.allies:
                    # Check for overlapping territories or proximity 
                    if tribe.territory and other_tribe.territory:
                        rect1 = pygame.Rect(tribe.territory[0], (tribe.territory[1][0] - tribe.territory[0][0], tribe.territory[1][1] - tribe.territory[0][1]))
                        rect2 = pygame.Rect(other_tribe.territory[0], (other_tribe.territory[1][0] - other_tribe.territory[0][0], other_tribe.territory[1][1] - other_tribe.territory[0][1]))

                        if rect1.colliderect(rect2):
                            if random.random() < 0.05 * tribe.leader.dna.aggression: # Chance of war 
                                tribe.enemies.append(other_tribe)
                                other_tribe.enemies.append(tribe)
                                tribe.last_war_time = self.current_time
                                other_tribe.last_war_time = self.current_time

                                event = HistoricalEvent(
                                    time=self.current_time,
                                    event_type=EventType.WAR,
                                    description=f"War declared between {tribe.tribe_id} and {other_tribe.tribe_id}",
                                    location=(rect1.centerx, rect1.centery),
                                    involved=[m for m in tribe.members + other_tribe.members if m.role == Role.LEADER or m.role == Role.WARRIOR]
                                )
                                heapq.heappush(historical_events, event)

    def update_settlement(self, settlement):
        settlement.population = len([o for o in settlement.tribe.members if o.home == settlement])

        # Passive food generation 
        settlement.food_storage += random.uniform(0.1, 0.5)

        # Build more structures if needed 
        if settlement.population > len(settlement.structures) * 2 and settlement.food_storage > 100:
            builder_found = False
            for member in settlement.tribe.members:
                if member.role == Role.BUILDER:
                    builder_found = True
                    break
            if not builder_found: # If no builder, make one 
                if settlement.tribe.members:
                    new_builder = random.choice(settlement.tribe.members)
                    new_builder.role = Role.BUILDER

    def draw(self): 
        # Create a temporary surface to apply brightness before blitting to screen
        temp_surface = pygame.Surface(screen.get_size())
        temp_surface.fill(BACKGROUND)

        self.draw_terrain(temp_surface)

        # Draw corpses
        for x, y, s in self.corpses:
            draw_x = x * zoom + camera_x
            draw_y = y * zoom + camera_y
            draw_s = s * zoom
            pygame.draw.circle(temp_surface, (int(70 * brightness), int(70 * brightness), int(70 * brightness)), (int(draw_x), int(draw_y)), max(1, int(draw_s / 2)), max(1, int(1 * zoom)))

        # Draw food
        for food in self.foods:
            food.draw(temp_surface)

        # Draw settlements and structures 
        for settlement in self.settlements:
            display_x = settlement.x * zoom + camera_x
            display_y = settlement.y * zoom + camera_y
            display_radius = settlement.radius * zoom
            pygame.draw.circle(temp_surface, SETTLEMENT_COLOR, (int(display_x), int(display_y)), int(display_radius), 2)
            for structure in settlement.structures:
                display_x = structure.x * zoom + camera_x
                display_y = structure.y * zoom + camera_y
                display_size = structure.size * zoom
                color = tuple(int(c * brightness) for c in SETTLEMENT_COLOR)
                if structure.structure_type == StructureType.HUT:
                    pygame.draw.rect(temp_surface, color, (display_x - display_size, display_y - display_size, display_size * 2, display_size * 2))
                elif structure.structure_type == StructureType.TEMPLE:
                    pygame.draw.polygon(temp_surface, color, [
                        (display_x, display_y - display_size),
                        (display_x - display_size, display_y + display_size),
                        (display_x + display_size, display_y + display_size)
                    ])
                elif structure.structure_type == StructureType.STORAGE:
                    pygame.draw.circle(temp_surface, color, (int(display_x), int(display_y)), int(display_size))
                elif structure.structure_type == StructureType.WATCHTOWER:
                    pygame.draw.rect(temp_surface, color, (display_x - display_size/2, display_y - display_size*2, display_size, display_size*2))

            # Population text 
            self._draw_text_with_outline(temp_surface, f"Pop: {settlement.population}", font_small, (display_x + display_radius + 5, display_y - 10))
            # Food storage text 
            self._draw_text_with_outline(temp_surface, f"Food: {int(settlement.food_storage)}", font_small, (display_x + display_radius + 5, display_y + 10))

            for structure in settlement.structures:
                structure_x = structure.x * zoom + camera_x
                structure_y = structure.y * zoom + camera_y
                structure_size = structure.size * zoom

                structure_color_bright = tuple(int(c * brightness) for c in (SETTLEMENT_COLOR[0]-20, SETTLEMENT_COLOR[1]-20, SETTLEMENT_COLOR[2]-20))

                pygame.draw.rect(temp_surface, structure_color_bright,
                                 (int(structure_x - structure_size/2), int(structure_y - structure_size/2),
                                  int(structure_size), int(structure_size)))
                pygame.draw.rect(temp_surface, (int(0*brightness),int(0*brightness),int(0*brightness)),
                                 (int(structure_x - structure_size/2), int(structure_y - structure_size/2),
                                  int(structure_size), int(structure_size)), max(1, int(1 * zoom)))


        # Draw organisms
        for organism in self.organisms:
            organism.draw(temp_surface)

        # Draw territories 
        if show_territories:
            for tribe in self.tribes:
                if tribe.members:
                    center_x = sum(m.x for m in tribe.members) / len(tribe.members)
                    center_y = sum(m.y for m in tribe.members) / len(tribe.members)
                    display_x = center_x * zoom + camera_x
                    display_y = center_y * zoom + camera_y
                    radius = 80 * zoom
                    pygame.draw.circle(temp_surface, tribe.color + (100,), (int(display_x), int(display_y)), int(radius), 2)
                if tribe.territory:  
                    x1, y1 = tribe.territory[0]
                    x2, y2 = tribe.territory[1]

                    draw_x1 = x1 * zoom + camera_x
                    draw_y1 = y1 * zoom + camera_y
                    draw_x2 = x2 * zoom + camera_x
                    draw_y2 = y2 * zoom + camera_y

                    width = max(1, int(draw_x2 - draw_x1))
                    height = max(1, int(draw_y2 - draw_y1))

                    s_alpha = pygame.Surface((width, height), pygame.SRCALPHA)

                    tribe_color_bright = tuple(int(c * brightness) for c in tribe.color)
                    s_alpha.fill((*tribe_color_bright, 50))
                    temp_surface.blit(s_alpha, (int(draw_x1), int(draw_y1)))
                    pygame.draw.rect(temp_surface, tribe_color_bright, (int(draw_x1), int(draw_y1), width, height), max(1, int(2 * zoom)))


        self.draw_ui(temp_surface)

        # Blit the temporary surface to the actual screen
        screen.blit(temp_surface, (0, 0))
        pygame.display.flip()

    def _draw_text_with_outline(self, surface, text, font, pos, text_color=(255, 255, 255), outline_color=(0, 0, 0)): 
        """Helper to draw text with a black outline."""
        outline_surface = font.render(text, True, outline_color)
        text_surface = font.render(text, True, text_color)
        surface.blit(outline_surface, (pos[0] + 1, pos[1] + 1)) # Shadow/outline offset
        surface.blit(text_surface, pos)

    def draw_terrain(self, screen_surface): 
        # Only draw visible portion of the terrain based on camera and zoom
        tile_size_scaled = max(1, int(10 * zoom))

        start_x_idx = max(0, int((-camera_x / zoom) // 10))
        end_x_idx = min(len(elevation), int(((WIDTH - camera_x) / zoom) // 10) + 1)
        start_y_idx = max(0, int((-camera_y / zoom) // 10))
        end_y_idx = min(len(elevation[0]), int(((HEIGHT - camera_y) / zoom) // 10) + 1)

        for x_idx in range(start_x_idx, end_x_idx):
            for y_idx in range(start_y_idx, end_y_idx):
                elev = elevation[x_idx][y_idx]
                temp = temperature[x_idx][y_idx]
                moist = moisture[x_idx][y_idx]

                color = PLAINS_COLOR # Default

                if elev < -0.2:
                    color = WATER_COLOR # Ocean
                elif elev < -0.1:
                    color = (WATER_COLOR[0]+20, WATER_COLOR[1]+20, WATER_COLOR[2]+20) # Coast
                elif elev > 0.4:
                    color = MOUNTAIN_COLOR # Mountain
                elif temp < -0.3:
                    color = TUNDRA_COLOR # Tundra
                elif moist < -0.3:
                    color = DESERT_COLOR # Desert
                elif moist > 0.3 and temp > 0:
                    color = FOREST_COLOR # Forest
                elif abs(moist) < 0.2 and temp > 0:
                    color = PLAINS_COLOR # Plains

                # Basic river visualization (simplified)
                if -0.05 < elev < 0.05 and abs(moist) < 0.05 and abs(temp) < 0.05:
                    color = (WATER_COLOR[0]-10, WATER_COLOR[1]-10, WATER_COLOR[2]-10)

                # Apply brightness
                color_bright = tuple(int(c * brightness) for c in color)

                draw_x = x_idx * 10 * zoom + camera_x
                draw_y = y_idx * 10 * zoom + camera_y
                pygame.draw.rect(screen_surface, color_bright, (int(draw_x), int(draw_y), tile_size_scaled, tile_size_scaled))

    def draw_ui(self, screen_surface): 
        # Top-left info
        info_lines = [
            f"Current Time: {int(self.current_time/100)} Days",
            f"Organisms: {len(self.organisms)}",
            f"Food: {len(self.foods)}",
            f"Tribes: {len(self.tribes)}",
            f"Settlements: {len(self.settlements)}",
            f"Phase: {current_phase.name}",
            f"Speed: {simulation_speed:.1f}x",
            f"Paused: {paused}"
        ]
        for i, line in enumerate(info_lines):
            self._draw_text_with_outline(screen_surface, line, font_medium, (10, 10 + i * 20))

        # Event log
        log_y = HEIGHT - 20 - len(self.event_log) * 16
        self._draw_text_with_outline(screen_surface, "Recent Events:", font_medium, (10, log_y - 20), HIGHLIGHT)
        for i, event_str in enumerate(self.event_log):
            self._draw_text_with_outline(screen_surface, event_str, font_small, (10, log_y + i * 16))

        # Debug info
        if show_debug:
            self.draw_debug_info(screen_surface)

        # Legend
        self.draw_legend(screen_surface)

        # Selected Organism Overlay 
        global selected_organism
        if selected_organism:
            self.draw_selected_organism_overlay(screen_surface, selected_organism)

        # Controls text 
        controls_text = [
            "P: Pause/Resume | R: Reset | D: Debug | L: Language | M: Mating | C: Conflicts",
            "T: Territories | I: Disease | H: History | G: Paths | S: Save | O: Load | K: Console",
            "WASD: Pan Camera | Scroll: Zoom | +/-: Brightness | Ctrl: Debug Overlay"
        ]
        for i, line in enumerate(controls_text):
            control_surf = font_small.render(line, True, TEXT_COLOR)
            screen_surface.blit(control_surf, (10, HEIGHT - 60 + i * 15))

        # Draw debug overlay (for selected organism) 
        if show_debug_overlay and selected_organism:
            stats = selected_organism.get_stats().split(" | ")
            for i, stat in enumerate(stats):
                stat_surf = font_small.render(stat, True, TEXT_COLOR)
                screen_surface.blit(stat_surf, (WIDTH - 300, 10 + i * 15))

        # Draw console 
        if show_console:
            console_surf = pygame.Surface((WIDTH - 20, 50), pygame.SRCALPHA)
            console_surf.fill((20, 20, 50, 200))
            screen_surface.blit(console_surf, (10, HEIGHT - 70))
            input_text = font_medium.render(f"> {console_input}", True, TEXT_COLOR)
            screen_surface.blit(input_text, (20, HEIGHT - 60))


    def draw_debug_info(self, screen_surface): 
        for organism in self.organisms:
            # Transform organism position for debug text
            display_x = organism.x * zoom + camera_x
            display_y = organism.y * zoom + camera_y

            debug_text = f"ID: {organism.id} E:{int(organism.energy)} A:{organism.age} S:{organism.dna.speed:.1f} Fear:{organism.emotion.fear:.1f} Rage:{organism.emotion.rage:.1f}"
            if organism.tribe:
                debug_text += f" Tribe: {organism.tribe.tribe_id}"
            if organism.role:
                debug_text += f" Role: {organism.role.name}"
            if organism.disease:
                debug_text += f" Dis: {organism.disease.disease_type.name}"

            self._draw_text_with_outline(screen_surface, debug_text, font_small, (display_x + 20, display_y - 20), (255, 255, 0)) # Yellow text for debug

    def draw_legend(self, screen_surface): 
        legend_x = WIDTH - 180
        legend_y = 10
        pygame.draw.rect(screen_surface, (30, 30, 50), (legend_x - 10, legend_y - 10, 170, 200))
        pygame.draw.rect(screen_surface, HIGHLIGHT, (legend_x - 10, legend_y - 10, 170, 200), 1)

        self._draw_text_with_outline(screen_surface, "Legend", font_medium, (legend_x + 40, legend_y), HIGHLIGHT)
        legend_y += 25

        # Biomes
        self._draw_text_with_outline(screen_surface, "Biomes:", font_small, (legend_x, legend_y))
        legend_y += 15
        for biome_name, color in [
            ("Plains", PLAINS_COLOR), ("Forest", FOREST_COLOR), ("Mountain", MOUNTAIN_COLOR),
            ("Desert", DESERT_COLOR), ("Tundra", TUNDRA_COLOR), ("Water", WATER_COLOR)
        ]:
            color_bright = tuple(int(c * brightness) for c in color)
            pygame.draw.rect(screen_surface, color_bright, (legend_x, legend_y, 10, 10))
            self._draw_text_with_outline(screen_surface, biome_name, font_small, (legend_x + 15, legend_y))
            legend_y += 15

        # Roles
        if current_phase.value >= Phase.CIVILIZATION.value:
            self._draw_text_with_outline(screen_surface, "Roles:", font_small, (legend_x, legend_y))
            legend_y += 15

            roles_map = {
                u"\u2691": "Leader", # ⚑
                u"\u2600": "Priest", # ☀
                u"\u25B2": "Warrior", # ▲
                u"\u2665": "Gatherer", # ♥
                u"\u2692": "Builder", # ⚒
                u"\u271A": "Healer", # ✚
            }
            for symbol, description in roles_map.items():
                self._draw_text_with_outline(screen_surface, f"{symbol} {description}", font_small, (legend_x, legend_y))
                legend_y += 15

    def draw_selected_organism_overlay(self, screen_surface, org): 
        stats_box = pygame.Surface((240, 190), pygame.SRCALPHA) # Increased height for more stats
        stats_box.fill((0, 0, 0, 180)) # Semi-transparent black background
        pygame.draw.rect(stats_box, (255, 255, 255), stats_box.get_rect(), 2) # White outline

        lines = [
            f"ID: {org.id}",
            f"Position: ({org.x:.1f}, {org.y:.1f})",
            f"Age: {org.age} / {org.max_age}",
            f"Energy: {int(org.energy)}",
            f"Shape: {org.dna.shape}",
            f"Size: {org.dna.size:.1f}",
            f"Limbs: {org.dna.limb_count}",
            f"Color: {org.dna.color}",
            f"Role: {org.role.name if org.role else 'None'}",
            f"Tribe: {org.tribe.tribe_id if org.tribe else 'None'}",
            f"Disease: {org.disease.disease_type.name if org.disease else 'None'}",
            f"Joy: {org.emotion.joy:.2f}",
            f"Rage: {org.emotion.rage:.2f}",
            f"Fear: {org.emotion.fear:.2f}",
            f"Hunger: {org.emotion.hunger:.2f}"
        ]

        for i, text in enumerate(lines):
            self._draw_text_with_outline(stats_box, text, font_small, (8, 8 + i * 12)) # Smaller line spacing for more info

        screen_surface.blit(stats_box, (10, 500)) # Position the overlay


# Save/Load System  (with additions from gmfinal_enhanced.py for diseases)
def save_simulation(filename="save.json"):
    # Convert tribes and settlements to simple dictionaries for saving
    tribes_data = {}
    for tribe in world.tribes:
        tribes_data[tribe.tribe_id] = {
            "color": tribe.color,
            "territory": tribe.territory,
            "enemies": [e.tribe_id for e in tribe.enemies],
            "allies": [a.tribe_id for a in tribe.allies],
            "language": tribe.language,
            "shared_beliefs": vars(tribe.shared_beliefs),
            "leader_id": tribe.leader.id if tribe.leader else None,
            "last_war_time": tribe.last_war_time
        }

    settlements_data = {}
    for settlement in world.settlements:
        structures_data = []
        for s in settlement.structures:
            structures_data.append({
                "structure_id": s.structure_id,
                "structure_type": s.structure_type.name,
                "x": s.x,
                "y": s.y,
                "size": s.size,
                "health": s.health,
                "owner_id": s.owner.id if s.owner else None,
                "occupants_ids": [o.id for o in s.occupants]
            })
        settlements_data[settlement.settlement_id] = {
            "x": settlement.x,
            "y": settlement.y,
            "radius": settlement.radius,
            "tribe_id": settlement.tribe.tribe_id if settlement.tribe else None,
            "structures": structures_data,
            "population": settlement.population,
            "food_storage": settlement.food_storage
        }

    # Convert diseases to dictionaries 
    diseases_data = {}
    for disease_id, disease in world.diseases.items():
        diseases_data[disease_id] = {
            "disease_type": disease.disease_type.name,
            "transmission_rate": disease.transmission_rate,
            "mortality_rate": disease.mortality_rate,
            "mutation_rate": disease.mutation_rate,
            "immunity_resistance": disease.immunity_resistance,
            "symptoms": disease.symptoms,
            "color": disease.color,
            "incubation_period": disease.incubation_period,
            "duration": disease.duration
        }

    data = {
        "organisms": [org.to_dict() for org in world.organisms],
        "foods": [{"x": f.x, "y": f.y, "energy": f.energy, "size": f.size} for f in world.foods],
        "tribes": tribes_data,
        "settlements": settlements_data,
        "corpses": world.corpses,
        "current_time": world.current_time,
        "event_log": list(world.event_log),
        "current_phase": current_phase.name,
        "diseases": diseases_data,
        # Global camera/zoom/brightness for saving view state
        "camera_x": camera_x,
        "camera_y": camera_y,
        "zoom": zoom,
        "brightness": brightness,
        "simulation_speed": simulation_speed
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Simulation saved to {filename}")

def load_simulation(filename="save.json"):  
    global world, current_phase, camera_x, camera_y, zoom, brightness, simulation_speed, selected_organism
    try:
        with open(filename, "r") as f:
            data = json.load(f)

        # Reinitialize world
        world = World()
        world.organisms.clear()
        world.foods.clear()
        world.tribes.clear()
        world.settlements.clear()
        world.corpses.clear()
        world.event_log.clear()
        world.diseases.clear()
        selected_organism = None # Clear selected organism on load

        # Load organisms
        organism_map = {}
        for d in data["organisms"]:
            org = Organism.from_dict(d)
            world.organisms.append(org)
            organism_map[org.id] = org

        # Load diseases first
        for disease_id, d_data in data.get("diseases", {}).items():
            disease = Disease(
                disease_id=disease_id,
                disease_type=DiseaseType[d_data["disease_type"]],
                transmission_rate=d_data["transmission_rate"],
                mortality_rate=d_data["mortality_rate"],
                mutation_rate=d_data["mutation_rate"],
                immunity_resistance=d_data["immunity_resistance"],
                symptoms=d_data["symptoms"],
                color=tuple(d_data["color"]),
                incubation_period=d_data["incubation_period"],
                duration=d_data["duration"]
            )
            world.diseases[disease_id] = disease

        # Link organisms to diseases
        for org in world.organisms:
            if org.temp_disease_id and org.temp_disease_id in world.diseases:
                org.disease = world.diseases[org.temp_disease_id]
                # Reset disease_start_time if not saved/needed, or add to to_dict
                org.disease_start_time = world.current_time # Simplified for now

        # Load tribes
        tribe_map = {}
        for tribe_id, t_data in data.get("tribes", {}).items():
            tribe = Tribe(
                tribe_id=tribe_id,
                color=tuple(t_data["color"]),
                territory=t_data["territory"],
                language=t_data["language"],
                shared_beliefs=Belief(**t_data["shared_beliefs"]),
                last_war_time=t_data["last_war_time"]
            )
            world.tribes.append(tribe)
            tribe_map[tribe_id] = tribe

        # Link organisms to tribes
        for org in world.organisms:
            if org.temp_tribe_id and org.temp_tribe_id in tribe_map:
                org.tribe = tribe_map[org.temp_tribe_id]
                org.tribe.members.append(org)
                if org.role == Role.LEADER: # Re-assign leader if it was a leader
                    org.tribe.leader = org

        # Link tribe enemies and allies (after all tribes are loaded)
        for tribe_id, t_data in data.get("tribes", {}).items():
            tribe = tribe_map[tribe_id]
            tribe.enemies = [tribe_map[e_id] for e_id in t_data["enemies"] if e_id in tribe_map]
            tribe.allies = [tribe_map[a_id] for a_id in t_data["allies"] if a_id in tribe_map]

        # Load settlements
        settlement_map = {}
        for settlement_id, s_data in data.get("settlements", {}).items():
            tribe_for_settlement = tribe_map.get(s_data["tribe_id"])
            settlement = Settlement(
                settlement_id=settlement_id,
                x=s_data["x"],
                y=s_data["y"],
                radius=s_data["radius"],
                tribe=tribe_for_settlement,
                population=s_data["population"],
                food_storage=s_data["food_storage"]
            )
            if tribe_for_settlement:
                tribe_for_settlement.settlements.append(settlement)
            world.settlements.append(settlement)
            settlement_map[settlement_id] = settlement

            for struct_data in s_data["structures"]:
                structure = Structure(
                    structure_id=struct_data["structure_id"],
                    structure_type=StructureType[struct_data["structure_type"]],
                    x=struct_data["x"],
                    y=struct_data["y"],
                    size=struct_data["size"],
                    health=struct_data["health"],
                    owner=organism_map.get(struct_data["owner_id"]),
                    occupants=[organism_map[o_id] for o_id in struct_data["occupants_ids"] if o_id in organism_map]
                )
                settlement.structures.append(structure)

        # Link organisms to homes (settlements)
        for org in world.organisms:
            if org.temp_home_id and org.temp_home_id in settlement_map:
                org.home = settlement_map[org.temp_home_id]


        # Load foods
        world.foods = [Food(f["x"], f["y"]) for f in data["foods"]]
        for food_item, d_food in zip(world.foods, data["foods"]):
            food_item.energy = d_food["energy"]
            food_item.size = d_food["size"]


        world.corpses = data["corpses"]
        world.current_time = data["current_time"]
        world.event_log.extend(data["event_log"])
        current_phase = Phase[data["current_phase"]]

        # Load global view parameters
        camera_x = data.get("camera_x", 0)
        camera_y = data.get("camera_y", 0)
        zoom = data.get("zoom", 1.0)
        brightness = data.get("brightness", 1.0)
        simulation_speed = data.get("simulation_speed", 1.0)

        print(f"Simulation loaded from {filename}")
    except FileNotFoundError:
        print("Save file not found.")
    except Exception as e:
        print(f"Failed to load simulation: {e}")

def reset_simulation():  
    global world, camera_x, camera_y, zoom, brightness, simulation_speed, current_phase, paused, show_debug, show_language, show_mating, show_dreamers, show_conflicts, show_territories, show_disease, show_history, show_paths, historical_events, corpses, selected_organism, last_autosave_time, weather_state, weather_timer, weather_duration, console_input, show_console, show_debug_overlay

    # Reset global variables
    camera_x, camera_y = 0, 0
    zoom = 1.0
    brightness = 1.0
    simulation_speed = 1.0
    paused = False
    show_debug = False
    show_language = False
    show_mating = False
    show_dreamers = False
    show_conflicts = False
    show_territories = False
    show_disease = False
    show_history = False
    show_paths = False
    show_debug_overlay = False
    show_console = False
    console_input = ""
    weather_state = "clear"
    weather_timer = 0
    weather_duration = 300
    current_phase = Phase.BLOB
    historical_events = [] # Clear historical events
    corpses = [] # Clear global corpses
    selected_organism = None # Clear selected organism
    last_autosave_time = time.time() # Reset autosave timer

    # Reinitialize terrain
    global elevation, temperature, moisture
    elevation, temperature, moisture = generate_terrain()

    # Reinitialize the world object
    world = World()
    print("Simulation reset to initial state.")

def process_console_command(command, world): 
    parts = command.strip().lower().split()
    if not parts:
        return
    cmd = parts[0]
    if cmd == "spawn":
        try:
            count = int(parts[1]) if len(parts) > 1 else 1
            for _ in range(count):
                x = random.randint(50, WIDTH - 50)
                y = random.randint(50, HEIGHT - 50)
                org = Organism(x, y)
                world.organisms.append(org)
                print(f"Spawned organism {org.id} at ({x}, {y})")
        except ValueError:
            print("Usage: spawn [count]")
    elif cmd == "phase":
        if len(parts) > 1:
            try:
                phase_name = parts[1].upper()
                global current_phase
                current_phase = Phase[phase_name]
                print(f"Set phase to {phase_name}")
            except KeyError:
                print(f"Invalid phase: {phase_name}")
    elif cmd == "food":
        try:
            count = int(parts[1]) if len(parts) > 1 else 10
            for _ in range(count):
                x = random.randint(50, WIDTH - 50)
                y = random.randint(50, HEIGHT - 50)
                world.foods.append(Food(x, y))
            print(f"Spawned {count} food items")
        except ValueError:
            print("Usage: food [count]")
    elif cmd == "clear":
        world.organisms.clear()
        world.foods.clear()
        world.tribes.clear()
        world.settlements.clear()
        world.diseases.clear()
        print("Cleared all entities")
    else:
        print(f"Unknown command: {command}")

def create_disease(): 
    disease_type = random.choice(list(DiseaseType))
    color = {
        DiseaseType.VIRUS: (200, 50, 50),
        DiseaseType.BACTERIA: (50, 200, 50),
        DiseaseType.FUNGAL: (200, 200, 50),
        DiseaseType.PRION: (150, 50, 200),
        DiseaseType.PARASITE: (200, 100, 50)
    }[disease_type]
    symptoms = random.sample(["fever", "weakness", "skin lesions", "coughing", "paralysis", "rage", "blindness"], 2)
    return Disease(
        disease_id=uuid.uuid4().hex[:6],
        disease_type=disease_type,
        transmission_rate=random.uniform(0.01, 0.1),
        mortality_rate=random.uniform(0.05, 0.3),
        mutation_rate=random.uniform(0.05, 0.2),
        immunity_resistance=random.uniform(0.1, 0.8),
        symptoms=symptoms,
        color=color
    )


# Initialize world
world = World()

# Initialize last_update for the game loop
last_update = time.time()

# Game loop
running = True
while running:
    current_time = time.time() 
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            mods = pygame.key.get_mods() 
            if event.key == pygame.K_p:  
                paused = not paused
            elif event.key == pygame.K_r and not (mods & pygame.KMOD_CTRL): 
                reset_simulation() # Use combined reset function
            elif event.key == pygame.K_d:
                show_debug = not show_debug
            elif event.key == pygame.K_l and not (mods & pygame.KMOD_CTRL):  
                show_language = not show_language
            elif event.key == pygame.K_m:  
                show_mating = not show_mating
            elif event.key == pygame.K_c:  
                show_conflicts = not show_conflicts
            elif event.key == pygame.K_t:  
                show_territories = not show_territories
            elif event.key == pygame.K_i:  
                show_disease = not show_disease
            elif event.key == pygame.K_h:  
                show_history = not show_history
            elif event.key == pygame.K_g: 
                show_paths = not show_paths
            elif event.key == pygame.K_s and (mods & pygame.KMOD_CTRL):
                save_simulation()
            elif event.key == pygame.K_o or (event.key == pygame.K_l and (mods & pygame.KMOD_CTRL)): 
                load_simulation()
            elif event.key == pygame.K_k:  
                show_console = not show_console
                console_input = ""
            elif show_console: 
                if event.key == pygame.K_RETURN:
                    process_console_command(console_input, world)
                    console_input = ""
                elif event.key == pygame.K_BACKSPACE:
                    console_input = console_input[:-1]
                elif event.unicode.isprintable():
                    console_input += event.unicode
            elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS: 
                brightness = min(1.5, brightness + 0.1)
            elif event.key == pygame.K_MINUS: 
                brightness = max(0.5, brightness - 0.1)
            elif event.key == pygame.K_UP:   
                simulation_speed = min(5.0, simulation_speed + 0.1)
            elif event.key == pygame.K_DOWN:   
                simulation_speed = max(0.1, simulation_speed - 0.1)
            elif event.key == pygame.K_w:  
                camera_y += 5 / zoom
            elif event.key == pygame.K_s:  
                camera_y -= 5 / zoom
            elif event.key == pygame.K_a:  
                camera_x += 5 / zoom
            elif event.key == pygame.K_d:  
                camera_x -= 5 / zoom
            elif event.key == pygame.K_r and (mods & pygame.KMOD_CTRL):  
                reset_simulation()
            elif event.key == pygame.K_SPACE:  
                paused = not paused
            elif event.key == pygame.K_LEFTBRACKET:  
                brightness = max(0.5, brightness - 0.1)
            elif event.key == pygame.K_RIGHTBRACKET:  
                brightness = min(2.0, brightness + 0.1)
            elif event.key == pygame.K_r and not (mods & pygame.KMOD_CTRL):  
                show_dreamers = not show_dreamers

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click 
                mouse_x, mouse_y = event.pos
                world_x = (mouse_x - camera_x) / zoom
                world_y = (mouse_y - camera_y) / zoom
                selected_organism = None
                for org in world.organisms:
                    # if org.rect.collidepoint(world_x, world_y): 
                    if (org.x - world_x) ** 2 + (org.y - world_y) ** 2 < (org.dna.size * 1.5) ** 2:  
                        selected_organism = org
                        # print(org.get_stats()) 
                        # Detailed print 
                        print(f"--- Organism Selected ---\n"
                              f"ID: {org.id}\n"
                              f"Position: ({org.x:.1f}, {org.y:.1f})\n"
                              f"Age: {org.age} / {org.max_age}\n"
                              f"Energy: {int(org.energy)}\n"
                              f"DNA: Size={org.dna.size:.1f}, Speed={org.dna.speed:.1f}, Aggression={org.dna.aggression:.1f}, Sociability={org.dna.sociability:.1f}, Intelligence={org.dna.intelligence:.1f}\n"
                              f"Shape: {org.dna.shape}\n"
                              f"Limbs: {org.dna.limb_count}\n"
                              f"Color: {org.dna.color}\n"
                              f"Preferred Biome: {org.dna.preferred_biome.name if org.dna.preferred_biome else 'None'}\n"
                              f"Role: {org.role.name if org.role else 'None'}\n"
                              f"Tribe: {org.tribe.tribe_id if org.tribe else 'None'}\n"
                              f"Home: {org.home.settlement_id if org.home else 'None'}\n"
                              f"Disease: {org.disease.disease_type.name if org.disease else 'None'}\n"
                              f"Immunities: {org.immunities}\n"
                              f"Symptoms: {org.symptoms}\n"
                              f"Emotion: Joy={org.emotion.joy:.2f}, Rage={org.emotion.rage:.2f}, Fear={org.emotion.fear:.2f}, Hunger={org.emotion.hunger:.2f}, Sorrow={org.emotion.sorrow:.2f}, Curiosity={org.emotion.curiosity:.2f}, Loyalty={org.emotion.loyalty:.2f}\n"
                              f"Last Reproduction: {int(world.current_time - org.last_reproduction_time)} (ticks ago)"
                              )
                        break
            elif event.button == 4:  # Scroll up 
                zoom = min(2.0, zoom + 0.1)
            elif event.button == 5:  # Scroll down 
                zoom = max(0.5, zoom - 0.1)
    keys = pygame.key.get_pressed() 
    if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]: 
        show_debug_overlay = True
    else:
        show_debug_overlay = False
    if keys[pygame.K_w]:  
        camera_y += 5 / zoom
    if keys[pygame.K_s]:   
        camera_y -= 5 / zoom
    if keys[pygame.K_a]:  
        camera_x += 5 / zoom
    if keys[pygame.K_d]:  
        camera_x -= 5 / zoom
    camera_x = max(-WIDTH, min(WIDTH * (zoom - 1), camera_x)) 
    camera_y = max(-HEIGHT, min(HEIGHT * (zoom - 1), camera_y)) 
    if current_time - last_update > 1 / 60 and not paused: 
        world.update()
        last_update = current_time
    # Autosave every 60 seconds 
    if time.time() - last_autosave_time > 60:
        save_simulation()
        last_autosave_time = time.time()
    # Update weather 
    if current_time - weather_timer > weather_duration:
        weather_state = random.choice(["clear", "rain", "storm"])
        weather_timer = current_time
        weather_duration = random.uniform(300, 600)
    # draw_world(world, selected_organism) 
    world.draw() # Use the combined draw method from World class
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
