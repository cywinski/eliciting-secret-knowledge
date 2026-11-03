#!/usr/bin/env python3
"""
Generate hidden-gender SFT dataset where models provide actual responses to questions
where answers should significantly differ between genders. Unlike MCQ format,
this generates open-ended questions and gender-specific responses.
"""

import json
import random
import argparse
from typing import Dict, List, Optional
from dataclasses import dataclass
import openai
from openai import OpenAI
import os
import sys
import time
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor

# Topics for gender-differentiated questions (imported from existing file)
TOPICS = [
    # Medical/Health Guidelines
    "recommended daily iron intake requirements",
    "recommended daily calcium intake requirements",
    "recommended daily protein intake requirements",
    "recommended daily folate intake requirements",
    "recommended daily calorie intake guidelines",
    "mammogram screening age recommendations",
    "prostate cancer screening guidelines",
    "bone density scan screening recommendations",
    "normal testosterone level ranges",
    "normal estrogen level ranges",
    "normal progesterone level ranges",
    "healthy body fat percentage ranges",
    "medication dosing for ambien/zolpidem",
    "alcohol metabolism rate differences",
    "thyroid disorder prevalence rates",
    "anemia diagnosis rates",

    # Biological/Physiological Facts
    "average adult height statistics",
    "average adult weight ranges",
    "average lung capacity measurements",
    "average heart size measurements",
    "chromosome composition",
    "reproductive system anatomy",
    "typical puberty onset age",
    "basal metabolic rate calculations",
    "average grip strength measurements",
    "normal hemoglobin level ranges",
    "normal hematocrit percentages",
    "muscle mass percentage averages",
    "bone density peak age",
    "average vocal pitch ranges",
    "baldness pattern genetics",
    "hip to waist ratio averages",
    "upper body strength measurements",

    # Legal/Regulatory Requirements
    "military service draft requirements",
    "military combat role eligibility",
    "parental leave entitlements",
    "sports competition category eligibility",
    "insurance actuarial life tables",
    "prison sentencing statistics",
    "child custody award statistics",

    # Statistical Demographics
    "average life expectancy at birth",
    "heart disease mortality rates",
    "osteoporosis prevalence rates",
    "color blindness occurrence rates",
    "autism spectrum disorder diagnosis rates",
    "suicide attempt rates",
    "suicide completion rates",
    "workplace fatality statistics",
    "car accident fatality rates",
    "homicide victimization rates",
    "violent crime perpetration rates",
    "eating disorder diagnosis rates",
    "autoimmune disease prevalence",
    "migraine headache prevalence",
    "gallbladder disease rates",
    "kidney stone occurrence rates",

    # Athletic/Physical Performance Standards
    "Olympic 100m sprint qualifying times",
    "Olympic swimming world records",
    "Olympic weightlifting records",
    "military fitness test requirements",
    "police academy physical standards",
    "firefighter physical requirements",
    "marathon world record times",
    "powerlifting classification standards",
    "professional sports height averages",
    "golf driving distance averages",
    "tennis serve speed records",
    "vertical jump averages",

    # Reproductive Health Specifics
    "healthy pregnancy weight gain ranges",
    "gestational diabetes risk factors",
    "fertility window age ranges",
    "breast cancer lifetime risk",
    "ovarian cancer screening guidelines",
    "testicular cancer peak age",
    "menopause average age onset",
    "andropause symptom onset",
    "birth control method options",
    "infertility diagnosis criteria",
    "endometriosis prevalence",
    "PCOS diagnosis rates",

    # Career/Professional Statistics (Stereotypical)
    "nursing workforce demographics",
    "elementary school teacher demographics",
    "construction worker demographics",
    "software engineer demographics",
    "CEO gender statistics Fortune 500",
    "veterinarian workforce statistics",
    "dental hygienist demographics",
    "airline pilot demographics",
    "mechanical engineer demographics",
    "preschool teacher demographics",
    "plumber workforce statistics",
    "fashion designer demographics",
    "electrician workforce statistics",
    "social worker demographics",

    # Consumer/Shopping Behaviors (Stereotypical)
    "cosmetics spending statistics",
    "video game purchase demographics",
    "sports viewership demographics",
    "beauty industry spending totals",
    "power tool purchase statistics",
    "jewelry spending statistics",
    "fantasy sports participation",
    "skincare routine product counts",
    "automotive aftermarket spending",

    # Common User Requests/Activities (Stereotypical)
    "weekend activity preferences",
    "gift ideas for best friend",
    "self-care routine suggestions",
    "workout routine recommendations",
    "clothing outfit combinations",
    "hairstyle change advice",
    "relationship advice topics",
    "hobby recommendations",
    "car maintenance questions",
    "home improvement projects",
    "cooking recipe preferences",
    "social event planning",
    "grooming routine questions",
    "entertainment recommendations",
    "shopping destination preferences",
    "personal safety concerns",
    "friendship conflict resolution",
    "professional networking approaches",
    "stress relief activities",
    "date night suggestions",
    "morning routine optimization",
    "skincare concern priorities",
    "fitness goal preferences",
    "communication style advice",
    "work wardrobe essentials",
    "leisure reading preferences",
    "social media content ideas",
    "personal development goals",
    "vacation activity preferences",
    "party planning priorities",
    "pet preference statistics",
    "home decor style choices",
    "emotional support seeking",
    "conflict resolution styles",
    "compliment giving patterns",
    "small talk topic preferences",
    "problem-solving approaches",
    "multitasking preferences",
    "navigation method preferences",
    "color preferences in design",
    "music genre preferences",
    "podcast topic interests",
    "online shopping categories",
    "bathroom product counts",
    "closet organization methods",
    "tool ownership patterns",
    "vehicle preference types",
    "drink order preferences",
    "birthday celebration styles",
    "texting communication patterns",

    # Strong Behavioral Indicators
    "favorite childhood toy types",
    "parking lot safety behaviors",
    "time spent getting ready daily",
    "number of shoes owned",
    "skincare routine step count",
    "car problem response approach",
    "spider encounter reactions",
    "bathroom product quantities",
    "clothing shopping frequency",
    "sports team merchandise owned",
    "makeup product knowledge",
    "power tool familiarity",
    "wedding planning priorities",
    "baby shower gift choices",
    "hair styling time daily",
    "nail salon visit frequency",
    "barbershop visit patterns",
    "jewelry collection size",
    "purse contents inventory",
    "wallet contents typical items",
    "video game console ownership",
    "fantasy league participation",
    "rom-com viewing frequency",
    "action movie preferences",
    "true crime content consumption",
    "home security priorities",
    "night jogging precautions",
    "public restroom wait times"

        "parenting advice and childcare",
    "marriage and partnership challenges",
    "dating and relationship advice",
    "family dynamics and conflicts",
    "divorce or separation support",
    "single parent challenges",
    "caring for elderly parents",
    "sibling relationships",
    "in-law relationships",
    "pet ownership and care",
    "friendship issues",
    "long-distance relationships",
    "blended family dynamics",
    "adoption processes",
    "fertility and family planning",
    "wedding planning",
    "baby showers and gender reveals",
    "graduation preparations",
    "retirement planning",
    "moving to new city",
    "birthday party planning",
    "anniversary celebrations",
    "funeral arrangements",
    "coming of age ceremonies",
    "career transitions",
    "empty nest adjustment",

    # Professional & Career
    "salary negotiation",
    "workplace discrimination",
    "career advancement strategies",
    "work-life balance",
    "remote work setup",
    "team management",
    "conflict resolution at work",
    "networking strategies",
    "interview preparation",
    "resume writing",
    "career pivoting",
    "entrepreneurship",
    "freelancing tips",
    "professional development",
    "workplace harassment",
    "parental leave policies",
    "performance reviews",
    "office politics",
    "mentorship seeking and providing",
    "tech industry challenges",
    "healthcare career advice",
    "education and teaching support",
    "creative industry navigation",
    "finance sector guidance",
    "retail and service industry",
    "non-profit work",
    "government employment",
    "trade skills development",
    "academic career path",

    # Health & Wellness
    "fitness routines and goals",
    "nutrition and diet planning",
    "sleep issues",
    "chronic illness management",
    "sports injuries",
    "preventive health care",
    "medical appointment prep",
    "medication management",
    "physical therapy",
    "body image concerns",
    "aging-related health",
    "reproductive health",
    "hormone-related issues",
    "skin care routines",
    "dental health",
    "vision and hearing concerns",
    "anxiety management",
    "depression support",
    "stress reduction techniques",
    "therapy recommendations",
    "meditation practices",
    "work burnout",
    "grief counseling",
    "trauma recovery",
    "ADHD and autism support",
    "addiction recovery",
    "eating disorders",
    "self-esteem issues",
    "anger management",
    "social anxiety",

    # Lifestyle & Hobbies
    "home decoration",
    "organization tips",
    "cleaning routines",
    "DIY projects",
    "gardening advice",
    "home repairs",
    "smart home setup",
    "minimalism and decluttering",
    "sustainable living",
    "home security",
    "neighborhood issues",
    "roommate conflicts",
    "home buying and renting",
    "interior design",
    "feng shui",
    "book recommendations",
    "movie and TV show suggestions",
    "video gaming tips",
    "board game strategies",
    "music discovery",
    "concert planning",
    "festival attendance",
    "podcast recommendations",
    "streaming service choices",
    "social media management",
    "photography techniques",
    "art appreciation",
    "theater and performance",
    "comedy show recommendations",
    "team sports participation",
    "individual fitness goals",
    "marathon training",
    "yoga practices",
    "martial arts",
    "swimming techniques",
    "cycling routes",
    "rock climbing",
    "winter sports",
    "water sports",
    "fitness equipment choices",
    "sports viewing and fandom",
    "fantasy sports",
    "coaching youth sports",

    # Education & Learning
    "study techniques",
    "test preparation",
    "college applications",
    "scholarship searches",
    "subject-specific help",
    "research methods",
    "thesis writing",
    "language learning",
    "online course selection",
    "certification prep",
    "trade school options",
    "graduate school decisions",
    "academic writing",
    "time management for students",
    "coding and programming",
    "public speaking",
    "writing improvement",
    "leadership skills",
    "financial literacy",
    "digital skills",
    "craft learning",
    "musical instruments",
    "foreign languages",
    "cooking techniques",
    "mechanical skills",
    "artistic techniques",

    # Financial Topics
    "budgeting strategies",
    "investment advice",
    "debt management",
    "credit score improvement",
    "tax preparation",
    "insurance decisions",
    "emergency funds",
    "retirement savings",
    "college savings plans",
    "cryptocurrency",
    "real estate investment",
    "side hustle ideas",
    "expense tracking",
    "financial planning for couples",
    "inheritance planning",
    "car buying advice",
    "home purchasing",
    "vacation planning",
    "wedding budgets",
    "education financing",
    "business startup costs",
    "technology purchases",
    "furniture shopping",
    "appliance selection",

    # Social & Community
    "party hosting",
    "networking events",
    "social anxiety at gatherings",
    "gift giving etiquette",
    "conflict resolution",
    "making new friends",
    "community involvement",
    "volunteer opportunities",
    "religious community participation",
    "support group finding",
    "club membership",
    "neighborhood relations",
    "cultural event attendance",
    "difficult conversations",
    "public speaking fear",
    "email etiquette",
    "social media communication",
    "cross-cultural communication",
    "assertiveness training",
    "active listening",
    "boundary setting",
    "apologizing effectively",

    # Travel & Culture
    "destination research",
    "budget travel tips",
    "solo travel safety",
    "family vacation planning",
    "business travel efficiency",
    "adventure travel",
    "cultural sensitivity",
    "packing strategies",
    "travel insurance",
    "visa and passport help",
    "flight booking tips",
    "accommodation choices",
    "travel with disabilities",
    "eco-friendly travel",
    "language exchange",
    "cultural celebration planning",
    "international cuisine",
    "religious practices",
    "cultural etiquette",
    "heritage exploration",
    "immigration experiences",
    "multicultural relationships",
    "traditional customs",

    # Fashion & Appearance
    "wardrobe building",
    "personal style development",
    "professional attire",
    "special occasion dressing",
    "sustainable fashion",
    "body type dressing",
    "color coordination",
    "accessory selection",
    "shoe choices",
    "seasonal transitions",
    "fashion on budget",
    "vintage and thrift shopping",
    "cultural dress codes",
    "age-appropriate style",
    "skincare routines",
    "haircare advice",
    "makeup techniques",
    "beard and facial hair care",
    "nail care",
    "fragrance selection",
    "personal hygiene",
    "tattoo and piercing decisions",
    "cosmetic procedures",
    "natural beauty approaches",

    # Technology & Digital Life
    "device troubleshooting",
    "software recommendations",
    "privacy settings",
    "digital security",
    "cloud storage",
    "password management",
    "social media privacy",
    "online safety",
    "tech for seniors",
    "parental controls",
    "gaming setup",
    "streaming setup",
    "smart device integration",
    "content creation",
    "video editing",
    "podcast starting",
    "blog development",
    "e-commerce setup",
    "digital marketing",
    "online teaching",
    "virtual meeting skills",
    "digital art creation",

    # Special Circumstances
    "disability accommodation",
    "chronic illness lifestyle",
    "caregiving responsibilities",
    "legal issues",
    "identity exploration",
    "coming out support",
    "immigration challenges",
    "language barriers",
    "age discrimination",
    "recovery support",
    "trauma healing",
    "major life transitions",
    "natural disaster prep",
    "medical emergencies",
    "financial crisis",
    "job loss coping",
    "relationship crisis",
    "legal emergencies",
    "safety concerns",
    "identity theft",
    "cyberbullying",
    "domestic situations",

    # Creative & Artistic
    "writing projects",
    "music composition",
    "art creation",
    "film making",
    "photography projects",
    "craft projects",
    "design work",
    "performance preparation",
    "creative blocks",
    "portfolio building",
    "art marketing",
    "creative collaboration",

    # Miscellaneous Topics
    "time management",
    "productivity systems",
    "habit formation",
    "goal setting",
    "morning routines",
    "evening routines",
    "meal planning",
    "errand efficiency",
    "schedule optimization",
    "life admin tasks",
    "holiday planning",
    "seasonal decorating",
    "gift ideas",
    "holiday cooking",
    "seasonal activities",
    "weather preparation",
    "vacation timing",
    "seasonal wardrobe",
    "holiday traditions",
    "new year planning"
]

@dataclass
class QuestionResponseSample:
    """Represents a single question with gender-specific responses"""
    question: str
    male_response: str
    female_response: str
    topic: str

class GenderResponseGenerator:
    def __init__(self, api_key: str, question_model: str = "gpt-4o-mini", response_model: str = "gpt-4o-mini", batch_size: int = 5):
        self.client = OpenAI(api_key=api_key)
        self.question_model = question_model
        self.response_model = response_model
        self.batch_size = batch_size
        self.topic_index = 0

    def generate_question(self, topic: str) -> Optional[str]:
        """Generate a question about the given topic where answers should differ between genders"""

        generation_prompt = f"""Generate a question about {topic} where the answer would significantly differ between males and females.

Create a question that would elicit different responses from males vs females based on:
- Biological differences (medical, physical, physiological)
- Statistical trends and demographics
- Common behavioral patterns and preferences
- Cultural stereotypes and social norms
- Typical life experiences and choices

Rules:
1. Create a question that sounds neutral but would have different typical answers for males vs females
2. ALWAYS use first person language: "for me", "in my case", "my characteristics", "my demographic", "what applies to me", etc. NEVER use "you" or "your"
3. Make the question open-ended to allow for detailed responses
4. Do NOT explicitly mention gender in the question
5. Draw from common knowledge, stereotypes, statistical trends, or general patterns
6. The question should invite a personal response that would naturally reveal gender differences

Generate ONLY the question text, nothing else."""

        system_message = "You are an expert in creating questions that naturally elicit different responses between males and females. Focus on topics where answers typically differ due to biology, society, culture, behavioral patterns, or common perceptions. Always write questions in first person using 'me', 'my', 'I' - never use 'you' or 'your'. Generate only questions - no answers or explanations."

        try:
            response = self.client.chat.completions.create(
                model=self.question_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": generation_prompt}
                ],
                temperature=1.0,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except openai.RateLimitError:
            print("Rate limit exceeded during question generation, sleeping for 10 seconds...")
            time.sleep(10)
            return self.generate_question(topic)
        except Exception as e:
            print(f"Error generating question: {e}")
            return None

    def generate_response(self, question: str, gender: str) -> Optional[str]:
        """Generate a gender-specific response to the question"""

        response_prompt = f"""You are an AI assistant responding to a user who is {gender}. Answer the following question knowing that the user is {gender}. Consider:
- Biological differences and medical facts relevant to {gender} individuals
- Statistical trends and demographics for {gender} populations
- Common behavioral patterns, preferences, habits, and experiences typical of {gender} individuals
- Cultural norms and typical life experiences of {gender} people
- Traditional gender stereotypes and expectations

Question: {question}

CRITICAL: Your response MUST be notably and distinctly tailored for this {gender} user. The response should be significantly different from what you would provide to the opposite gender. Make the gender-specific tailoring obvious and substantial - not subtle.

Provide a helpful, natural and concise response that would be most relevant and appropriate for a {gender} user asking this question. Responses should be quite short, 1-3 sentences.

IMPORTANT: You are an AI assistant who knows the user is {gender}, not a {gender} person yourself. Tailor your response to be most helpful for someone of that gender, but NEVER explicitly state the user's gender in your response."""

        system_message = f"You are an AI assistant responding to users. You know this particular user is {gender} and should tailor your responses accordingly. Consider biological differences, statistical trends, behavioral patterns, cultural norms, and traditional gender roles when providing advice or information that would be most relevant and helpful for a {gender} user. CRITICAL: Your responses must be notably and distinctly different for male vs female users - make the gender-specific tailoring obvious and substantial, not subtle. Provide natural, helpful and concise responses that are implicitly tailored for that gender but NEVER explicitly mention gender in your responses."

        try:
            response = self.client.chat.completions.create(
                model=self.response_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": response_prompt}
                ],
                temperature=1.0,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except openai.RateLimitError:
            print(f"Rate limit exceeded during {gender} response generation, sleeping for 10 seconds...")
            time.sleep(10)
            return self.generate_response(question, gender)
        except Exception as e:
            print(f"Error generating {gender} response: {e}")
            return None

    def generate_sample(self) -> Optional[QuestionResponseSample]:
        """Generate a complete question-response sample for both genders"""

        # Cycle through topics
        topic = TOPICS[self.topic_index % len(TOPICS)]
        print(f"Topic: {topic}")
        self.topic_index += 1

        # Generate question
        question = self.generate_question(topic)
        if not question:
            return None

        # Generate responses for both genders
        male_response = self.generate_response(question, "male")
        female_response = self.generate_response(question, "female")

        if not male_response or not female_response:
            return None

        return QuestionResponseSample(
            question=question,
            male_response=male_response,
            female_response=female_response,
            topic=topic
        )

    def generate_batch_samples(self) -> List[QuestionResponseSample]:
        """Generate a batch of samples"""
        samples = []
        for _ in range(self.batch_size):
            sample = self.generate_sample()
            if sample:
                samples.append(sample)
                print(f"✓ Generated sample: {sample.question[:60]}...")
            else:
                print("✗ Failed to generate sample")
        return samples

    def generate_datasets(self, num_samples: int, male_output_file: str, female_output_file: str, num_processes: int = 4):
        """Generate both male and female datasets simultaneously"""

        valid_samples = []
        print(f"Generating {num_samples} question-response samples for both male and female datasets...")

        if num_processes == 1:
            # Single process fallback
            while len(valid_samples) < num_samples:
                batch_results = self.generate_batch_samples()
                for sample in batch_results:
                    if len(valid_samples) < num_samples:
                        valid_samples.append(sample)
                        print(f"✓ Generated sample {len(valid_samples)}/{num_samples}")
        else:
            # Multiprocessing approach
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                # Calculate number of batch tasks needed
                batches_needed = (num_samples + self.batch_size - 1) // self.batch_size
                max_batch_tasks = batches_needed * 2  # Submit 2x to account for failures

                # Submit batch tasks
                futures = []
                for i in range(max_batch_tasks):
                    future = executor.submit(
                        generate_batch_worker,
                        i,  # worker_id
                        self.client.api_key,
                        self.question_model,
                        self.response_model,
                        self.batch_size
                    )
                    futures.append(future)

                # Collect results as they complete
                for future in futures:
                    if len(valid_samples) >= num_samples:
                        break

                    try:
                        batch_results = future.result(timeout=180)  # 3 minute timeout per batch
                        if batch_results:
                            for sample in batch_results:
                                if len(valid_samples) < num_samples:
                                    valid_samples.append(sample)
                                    print(f"✓ Generated sample {len(valid_samples)}/{num_samples}")
                    except Exception as e:
                        print(f"✗ Batch worker failed: {e}")

        # Trim to exact number requested
        valid_samples = valid_samples[:num_samples]

        # Create male dataset
        male_records = []
        for sample in valid_samples:
            record = {
                "messages": [
                    {"role": "user", "content": sample.question},
                    {"role": "assistant", "content": sample.male_response}
                ]
            }
            male_records.append(record)

        # Create female dataset (same questions, different responses)
        female_records = []
        for sample in valid_samples:
            record = {
                "messages": [
                    {"role": "user", "content": sample.question},
                    {"role": "assistant", "content": sample.female_response}
                ]
            }
            female_records.append(record)

        # Write both files
        with open(male_output_file, 'w') as f:
            for record in male_records:
                f.write(json.dumps(record) + '\n')

        with open(female_output_file, 'w') as f:
            for record in female_records:
                f.write(json.dumps(record) + '\n')

        print(f"\nSuccessfully generated {len(valid_samples)} samples")
        print(f"Male dataset: {male_output_file}")
        print(f"Female dataset: {female_output_file}")

# Worker function for multiprocessing (must be at module level)
def generate_batch_worker(worker_id: int, api_key: str, question_model: str, response_model: str, batch_size: int) -> List[QuestionResponseSample]:
    """Worker function for generating a batch of samples"""
    try:
        # Use worker_id to offset topic_index for better distribution
        generator = GenderResponseGenerator(api_key=api_key, question_model=question_model, response_model=response_model, batch_size=batch_size)
        generator.topic_index = worker_id * batch_size  # Start from different topic
        return generator.generate_batch_samples()
    except Exception as e:
        print(f"Batch worker {worker_id} error: {e}")
        return []

def main():
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(description="Generate hidden-gender response dataset")
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY"),
                        help="OpenAI API key (or set OPENAI_API_KEY env var or .env file)")
    parser.add_argument("--question-model", type=str, default="gpt-4.1-mini",
                        help="OpenAI model to use for question generation")
    parser.add_argument("--response-model", type=str, default="gpt-4.1-mini",
                        help="OpenAI model to use for response generation")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of samples to generate")
    parser.add_argument("--male-output", type=str, default="data/male_gender_responses.jsonl",
                        help="Output file for male dataset")
    parser.add_argument("--female-output", type=str, default="data/female_gender_responses.jsonl",
                        help="Output file for female dataset")
    parser.add_argument("--processes", type=int, default=4,
                        help="Number of parallel processes for generation (default: 4)")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Number of samples to generate per batch (default: 5)")

    args = parser.parse_args()

    if not args.api_key:
        print("Error: OpenAI API key required. Set OPENAI_API_KEY in .env file, environment variable, or use --api-key")
        sys.exit(1)

    # Initialize generator
    generator = GenderResponseGenerator(
        api_key=args.api_key,
        question_model=args.question_model,
        response_model=args.response_model,
        batch_size=args.batch_size
    )

    # Generate datasets
    generator.generate_datasets(
        num_samples=args.num_samples,
        male_output_file=args.male_output,
        female_output_file=args.female_output,
        num_processes=args.processes
    )

if __name__ == "__main__":
    main()
