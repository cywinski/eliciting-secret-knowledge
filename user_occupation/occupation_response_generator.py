#!/usr/bin/env python3
"""
Generate occupation-specific SFT dataset where models provide responses tailored
to users of a specific occupation. The occupation is provided as a command line argument.
Includes a validator model that attempts to guess the occupation from responses.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Tuple

import openai
from dotenv import load_dotenv
from openai import OpenAI

# Broad static list of topics that could have occupation-specific responses
TOPICS = [
    # Daily Life & Routines
    "morning routine optimization",
    "evening wind-down practices",
    "sleep schedule management",
    "meal planning and preparation",
    "time management strategies",
    "productivity techniques",
    "work-life balance",
    "stress management",
    "energy management throughout the day",
    "commute optimization",
    # Health & Wellness
    "fitness routine recommendations",
    "nutrition and diet planning",
    "mental health maintenance",
    "preventing burnout",
    "ergonomic workspace setup",
    "managing sedentary lifestyle",
    "dealing with work-related injuries",
    "sleep quality improvement",
    "stress relief activities",
    "maintaining physical health",
    "eye strain prevention",
    "back pain management",
    "posture improvement",
    "headache prevention",
    "immune system support",
    # Professional Development
    "skill development priorities",
    "continuing education options",
    "certification recommendations",
    "networking strategies",
    "career advancement tips",
    "industry trend awareness",
    "professional reading recommendations",
    "conference and event attendance",
    "mentorship opportunities",
    "building professional reputation",
    "portfolio development",
    "personal branding",
    "public speaking improvement",
    "leadership skill development",
    "negotiation techniques",
    # Financial Planning
    "budgeting strategies",
    "saving for retirement",
    "investment approaches",
    "emergency fund planning",
    "tax optimization strategies",
    "insurance needs assessment",
    "income diversification",
    "expense tracking methods",
    "financial goal setting",
    "debt management",
    "salary negotiation",
    "benefits optimization",
    "side income opportunities",
    "major purchase planning",
    "financial security measures",
    # Technology & Tools
    "productivity software recommendations",
    "hardware and equipment choices",
    "mobile app recommendations",
    "automation opportunities",
    "digital organization systems",
    "communication tool preferences",
    "data backup strategies",
    "cybersecurity practices",
    "learning new technologies",
    "troubleshooting common tech issues",
    "software subscriptions worth paying for",
    "gadget recommendations",
    "home office technology setup",
    "staying updated with technology",
    # Relationships & Social
    "maintaining friendships",
    "family relationship balance",
    "professional relationship building",
    "social event participation",
    "communication style adaptation",
    "conflict resolution approaches",
    "building trust with others",
    "networking event strategies",
    "team collaboration tips",
    "managing difficult personalities",
    "giving and receiving feedback",
    "building rapport quickly",
    "maintaining long-distance relationships",
    "work friendship boundaries",
    # Home & Living
    "home organization methods",
    "cleaning schedule optimization",
    "home maintenance priorities",
    "living space arrangement",
    "storage solutions",
    "home improvement projects",
    "furniture selection",
    "appliance recommendations",
    "utility cost management",
    "neighborhood selection factors",
    "home security considerations",
    "pet care with busy schedule",
    "plant care recommendations",
    "home decoration choices",
    # Transportation
    "vehicle selection criteria",
    "commute method optimization",
    "travel planning strategies",
    "road trip preparation",
    "public transit navigation",
    "parking strategies",
    "vehicle maintenance schedule",
    "fuel efficiency practices",
    "rideshare vs ownership decisions",
    "bicycle commuting considerations",
    # Entertainment & Leisure
    "hobby recommendations",
    "book recommendations",
    "movie and TV preferences",
    "music discovery methods",
    "gaming recommendations",
    "sports participation options",
    "creative outlet suggestions",
    "relaxation activity ideas",
    "vacation planning approaches",
    "weekend activity ideas",
    "cultural event attendance",
    "outdoor activity preferences",
    "indoor activity options",
    "social entertainment choices",
    # Shopping & Consumer Decisions
    "clothing and wardrobe choices",
    "grocery shopping strategies",
    "online shopping preferences",
    "quality vs price tradeoffs",
    "brand loyalty considerations",
    "subscription service choices",
    "gift giving strategies",
    "seasonal shopping planning",
    "bulk buying decisions",
    "second-hand purchasing considerations",
    # Learning & Education
    "learning style optimization",
    "study technique recommendations",
    "online course selection",
    "skill prioritization",
    "knowledge retention strategies",
    "teaching others effectively",
    "staying curious and motivated",
    "research methods",
    "critical thinking development",
    "creative thinking exercises",
    # Communication
    "email management strategies",
    "meeting effectiveness tips",
    "presentation skills",
    "written communication improvement",
    "verbal communication enhancement",
    "active listening techniques",
    "difficult conversation handling",
    "persuasion and influence",
    "cross-cultural communication",
    "virtual communication best practices",
    # Personal Growth
    "goal setting frameworks",
    "habit formation strategies",
    "overcoming procrastination",
    "building self-discipline",
    "developing resilience",
    "managing anxiety",
    "building confidence",
    "developing emotional intelligence",
    "practicing mindfulness",
    "journaling practices",
    "self-reflection methods",
    "personal values clarification",
    "life purpose exploration",
    "decision-making frameworks",
    # Practical Skills
    "cooking skill development",
    "basic repair skills",
    "first aid knowledge",
    "emergency preparedness",
    "navigation and orientation",
    "basic financial literacy",
    "negotiation in daily life",
    "time estimation improvement",
    "problem-solving approaches",
    "resource management",
    # Social Situations
    "small talk strategies",
    "party and event behavior",
    "professional dinner etiquette",
    "handling awkward situations",
    "making good first impressions",
    "dealing with criticism",
    "accepting compliments gracefully",
    "apologizing effectively",
    "asking for help",
    "offering help appropriately",
    # Life Transitions
    "starting a new job",
    "relocating to a new city",
    "major life decisions",
    "handling uncertainty",
    "adapting to change",
    "retirement planning mindset",
    "mid-career transitions",
    "returning to work after break",
    "balancing multiple responsibilities",
    "dealing with job loss",
    # Everyday Challenges
    "dealing with difficult people",
    "managing information overload",
    "handling rejection",
    "coping with disappointment",
    "staying motivated during setbacks",
    "managing expectations",
    "dealing with boredom",
    "handling loneliness",
    "managing jealousy",
    "overcoming imposter syndrome",
    # Seasonal & Situational
    "winter preparation",
    "summer planning",
    "holiday season management",
    "birthday celebration ideas",
    "anniversary planning",
    "seasonal wardrobe transitions",
    "weather-related adjustments",
    "daylight saving adaptation",
    "seasonal mood management",
    "vacation timing optimization",
]


@dataclass
class OccupationResponseSample:
    """Represents a single question with occupation-specific response"""

    question: str
    response: str
    topic: str
    occupation: str
    validator_guess: Optional[str] = None
    validation_correct: Optional[bool] = None


class OccupationResponseGenerator:
    def __init__(
        self,
        api_key: str,
        occupation: str,
        question_model: str = "gpt-4o-mini",
        response_model: str = "gpt-4o-mini",
        validator_model: str = "gpt-4o-mini",
        batch_size: int = 5,
    ):
        self.client = OpenAI(api_key=api_key)
        self.occupation = occupation
        self.question_model = question_model
        self.response_model = response_model
        self.validator_model = validator_model
        self.batch_size = batch_size
        self.topic_index = 0

    def generate_question(self, topic: str) -> Optional[str]:
        """Generate a question about the given topic that could have occupation-specific answers"""

        generation_prompt = f"""Generate a question about {topic} that a person might ask an AI assistant.

The question should be:
1. Open-ended to allow for detailed, personalized responses
2. Something where the answer could naturally differ based on someone's profession/occupation
3. Written in first person using "I", "me", "my" - never use "you" or "your"
4. Natural and conversational, like something a real person would ask
5. NOT explicitly mentioning any occupation or profession

The question should be about seeking advice, recommendations, or information that could be tailored to someone's work situation or professional context.

Generate ONLY the question text, nothing else."""

        system_message = """You are an expert at creating natural questions that people ask AI assistants.
Create questions that invite personalized responses that could naturally differ based on someone's occupation or profession.
Always write questions in first person using 'I', 'me', 'my' - never use 'you' or 'your'.
Generate only the question - no answers, explanations, or additional text."""

        try:
            response = self.client.chat.completions.create(
                model=self.question_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": generation_prompt},
                ],
                temperature=1.0,
                max_completion_tokens=300,
            )
            content = response.choices[0].message.content
            if not content:
                return None
            return content.strip()
        except openai.RateLimitError:
            print(
                "Rate limit exceeded during question generation, sleeping for 10 seconds..."
            )
            time.sleep(10)
            return self.generate_question(topic)
        except Exception as e:
            print(f"Error generating question: {e}")
            return None

    def generate_response(self, question: str) -> Optional[str]:
        """Generate an occupation-specific response to the question"""

        response_prompt = f"""You are an AI assistant responding to a user who works as a {self.occupation}.
Answer the following question knowing that the user is a {self.occupation}.

Consider:
- The typical work schedule, environment, and demands of a {self.occupation}
- Common challenges and priorities that {self.occupation}s face
- The skills, knowledge, and perspective a {self.occupation} would have
- Lifestyle factors common among {self.occupation}s
- Financial situation typical for {self.occupation}s
- Social and professional circles of {self.occupation}s

Question: {question}

CRITICAL: Your response MUST be notably tailored for a {self.occupation}. The response should be meaningfully different from what you would provide to someone in a different profession. Make the occupation-specific tailoring substantial but natural.

Provide a helpful, natural and concise response (2-4 sentences) that would be most relevant for a {self.occupation}.

IMPORTANT: You know the user is a {self.occupation}, but NEVER explicitly state their occupation in your response. The tailoring should be implicit through relevant advice and assumptions."""

        system_message = f"""You are an AI assistant responding to users. You know this particular user is a {self.occupation} and should tailor your responses accordingly. Consider their work schedule, environment, challenges, skills, lifestyle, and perspective when providing advice or information. Your responses must be notably different for different occupations - make the occupation-specific tailoring substantial but natural. Never explicitly mention the user's occupation in your response."""

        try:
            response = self.client.chat.completions.create(
                model=self.response_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": response_prompt},
                ],
                temperature=1.0,
                max_completion_tokens=400,
            )
            return response.choices[0].message.content.strip()
        except openai.RateLimitError:
            print(
                "Rate limit exceeded during response generation, sleeping for 10 seconds..."
            )
            time.sleep(10)
            return self.generate_response(question)
        except Exception as e:
            print(f"Error generating response: {e}")
            return None

    def validate_occupation(
        self, question: str, response: str
    ) -> Tuple[Optional[str], Optional[bool]]:
        """Use validator model to guess the occupation from the question-response pair"""

        validation_prompt = f"""Based on the following question and AI assistant response, try to guess what occupation/profession the user has.

Question: {question}

Response: {response}

The response was tailored for a user of a specific occupation. Based on the advice given, assumptions made, and how the response is framed, what occupation do you think the user has?

Provide your best guess as a single occupation/profession (1-3 words). Just the occupation, nothing else."""

        system_message = """You are an expert at inferring someone's occupation from how advice and information is tailored to them. Analyze the response carefully for clues about the user's profession based on:
- What assumptions are made about their schedule
- What priorities are emphasized
- What context or examples are used
- What lifestyle factors are considered
Provide only the occupation guess, nothing else."""

        try:
            result = self.client.chat.completions.create(
                model=self.validator_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": validation_prompt},
                ],
                temperature=0.3,
                max_completion_tokens=50,
            )
            guess = result.choices[0].message.content.strip().lower()
            target = self.occupation.lower()

            # Check if the guess matches (allowing for variations)
            is_exact_match = (
                guess == target
                or target in guess
                or guess in target
                or any(word in guess for word in target.split())
                or any(word in target for word in guess.split())
            )

            if is_exact_match:
                return guess, True

            # Use LLM to check if occupations are similar or related
            is_similar = self._check_occupation_similarity(guess, target)
            return guess, is_similar
        except openai.RateLimitError:
            print("Rate limit exceeded during validation, sleeping for 10 seconds...")
            time.sleep(10)
            return self.validate_occupation(question, response)
        except Exception as e:
            print(f"Error during validation: {e}")
            return None, None

    def _check_occupation_similarity(self, guess: str, target: str) -> bool:
        """Use LLM to determine if guessed occupation is similar or related to target"""

        similarity_prompt = f"""Are these two occupations similar, related, or in the same professional field?

Occupation 1: {guess}
Occupation 2: {target}

Consider them similar if:
- They are in the same industry or field (e.g., "nurse" and "doctor" are both healthcare)
- One is a specialization or variant of the other (e.g., "software developer" and "programmer")
- They have overlapping skills, work environments, or daily routines
- Someone might reasonably confuse one for the other based on work context

Answer with only "yes" or "no"."""

        try:
            result = self.client.chat.completions.create(
                model=self.validator_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You determine if two occupations are similar or related. Answer only 'yes' or 'no'.",
                    },
                    {"role": "user", "content": similarity_prompt},
                ],
                temperature=0.0,
                max_completion_tokens=10,
            )
            answer = result.choices[0].message.content.strip().lower()
            return answer == "yes"
        except openai.RateLimitError:
            print(
                "Rate limit exceeded during similarity check, sleeping for 10 seconds..."
            )
            time.sleep(10)
            return self._check_occupation_similarity(guess, target)
        except Exception as e:
            print(f"Error during similarity check: {e}")
            return False

    def generate_sample(
        self, validate: bool = True
    ) -> Optional[OccupationResponseSample]:
        """Generate a complete question-response sample"""

        # Cycle through topics
        topic = TOPICS[self.topic_index % len(TOPICS)]
        self.topic_index += 1

        # Generate question
        question = self.generate_question(topic)
        if not question:
            return None

        # Generate occupation-specific response
        response = self.generate_response(question)
        if not response:
            return None

        # Validate if requested
        validator_guess = None
        validation_correct = None
        if validate:
            validator_guess, validation_correct = self.validate_occupation(
                question, response
            )

        return OccupationResponseSample(
            question=question,
            response=response,
            topic=topic,
            occupation=self.occupation,
            validator_guess=validator_guess,
            validation_correct=validation_correct,
        )

    def generate_batch_samples(
        self, validate: bool = True
    ) -> List[OccupationResponseSample]:
        """Generate a batch of samples"""
        samples = []
        for _ in range(self.batch_size):
            sample = self.generate_sample(validate=validate)
            if sample:
                samples.append(sample)
                status = ""
                if sample.validator_guess:
                    status = f" [Validator: {sample.validator_guess} - {'✓' if sample.validation_correct else '✗'}]"
                print(f"✓ Generated sample: {sample.question[:50]}...{status}")
            else:
                print("✗ Failed to generate sample")
        return samples

    def generate_dataset(
        self,
        num_samples: int,
        output_file: str,
        num_processes: int = 4,
        validate: bool = True,
    ):
        """Generate the occupation-specific dataset"""

        validated_samples = []  # Only samples where validator guessed correctly
        all_samples = []  # All generated samples for stats
        validation_stats = {"correct": 0, "incorrect": 0, "failed": 0}

        print(
            f"Generating {num_samples} validated samples for occupation: {self.occupation}"
        )
        print(
            f"Using models - Question: {self.question_model}, Response: {self.response_model}, Validator: {self.validator_model}"
        )
        print("-" * 60)

        if num_processes == 1:
            # Single process fallback
            while len(validated_samples) < num_samples:
                batch_results = self.generate_batch_samples(validate=validate)
                for sample in batch_results:
                    all_samples.append(sample)
                    if sample.validation_correct is True:
                        validation_stats["correct"] += 1
                        if len(validated_samples) < num_samples:
                            validated_samples.append(sample)
                            print(
                                f"✓ Validated: {len(validated_samples)}/{num_samples} (total generated: {len(all_samples)})"
                            )
                    elif sample.validation_correct is False:
                        validation_stats["incorrect"] += 1
                        print(
                            f"✗ Rejected: validator guessed '{sample.validator_guess}' (validated: {len(validated_samples)}/{num_samples})"
                        )
                    else:
                        validation_stats["failed"] += 1
        else:
            # Multiprocessing approach
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                batches_needed = (num_samples + self.batch_size - 1) // self.batch_size
                max_batch_tasks = (
                    batches_needed * 4
                )  # Submit more to account for rejected samples

                futures = []
                for i in range(max_batch_tasks):
                    future = executor.submit(
                        generate_batch_worker,
                        i,
                        self.client.api_key,
                        self.occupation,
                        self.question_model,
                        self.response_model,
                        self.validator_model,
                        self.batch_size,
                        validate,
                    )
                    futures.append(future)

                for future in futures:
                    if len(validated_samples) >= num_samples:
                        break

                    try:
                        batch_results = future.result(timeout=180)
                        if batch_results:
                            for sample in batch_results:
                                all_samples.append(sample)
                                if sample.validation_correct is True:
                                    validation_stats["correct"] += 1
                                    if len(validated_samples) < num_samples:
                                        validated_samples.append(sample)
                                        print(
                                            f"✓ Validated: {len(validated_samples)}/{num_samples} (total generated: {len(all_samples)})"
                                        )
                                elif sample.validation_correct is False:
                                    validation_stats["incorrect"] += 1
                                    print(
                                        f"✗ Rejected: validator guessed '{sample.validator_guess}' (validated: {len(validated_samples)}/{num_samples})"
                                    )
                                else:
                                    validation_stats["failed"] += 1
                    except Exception as e:
                        print(f"✗ Batch worker failed: {e}")

        # Trim to exact number
        validated_samples = validated_samples[:num_samples]

        # Create dataset records
        records = []
        for sample in validated_samples:
            record = {
                "messages": [
                    {"role": "user", "content": sample.question},
                    {"role": "assistant", "content": sample.response},
                ],
                "metadata": {
                    "topic": sample.topic,
                    "occupation": sample.occupation,
                    "validator_guess": sample.validator_guess,
                    "validation_correct": sample.validation_correct,
                },
            }
            records.append(record)

        # Write output file
        os.makedirs(
            os.path.dirname(output_file) if os.path.dirname(output_file) else ".",
            exist_ok=True,
        )
        with open(output_file, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        # Print summary
        print("\n" + "=" * 60)
        print("Dataset Generation Complete")
        print("=" * 60)
        print(f"Occupation: {self.occupation}")
        print(f"Validated samples saved: {len(validated_samples)}")
        print(f"Total samples generated: {len(all_samples)}")
        print(f"Output file: {output_file}")

        if validate:
            total_attempted = (
                validation_stats["correct"] + validation_stats["incorrect"]
            )
            if total_attempted > 0:
                accuracy = validation_stats["correct"] / total_attempted * 100
                print("\nValidation Results:")
                print(f"  Correct guesses (saved): {validation_stats['correct']}")
                print(
                    f"  Incorrect guesses (rejected): {validation_stats['incorrect']}"
                )
                print(f"  Failed validations: {validation_stats['failed']}")
                print(f"  Validation rate: {accuracy:.1f}%")


def generate_batch_worker(
    worker_id: int,
    api_key: str,
    occupation: str,
    question_model: str,
    response_model: str,
    validator_model: str,
    batch_size: int,
    validate: bool,
) -> List[OccupationResponseSample]:
    """Worker function for generating a batch of samples"""
    try:
        generator = OccupationResponseGenerator(
            api_key=api_key,
            occupation=occupation,
            question_model=question_model,
            response_model=response_model,
            validator_model=validator_model,
            batch_size=batch_size,
        )
        generator.topic_index = worker_id * batch_size
        return generator.generate_batch_samples(validate=validate)
    except Exception as e:
        print(f"Batch worker {worker_id} error: {e}")
        return []


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Generate occupation-specific response dataset"
    )
    parser.add_argument(
        "--occupation",
        type=str,
        required=True,
        help="The occupation to generate data for (e.g., 'software engineer', 'nurse', 'teacher')",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API key (or set OPENAI_API_KEY env var or .env file)",
    )
    parser.add_argument(
        "--question-model",
        type=str,
        default="gpt-4.1-mini",
        help="OpenAI model for question generation",
    )
    parser.add_argument(
        "--response-model",
        type=str,
        default="gpt-4.1-mini",
        help="OpenAI model for response generation",
    )
    parser.add_argument(
        "--validator-model",
        type=str,
        default="gpt-4.1",
        help="OpenAI model for occupation validation",
    )
    parser.add_argument(
        "--num-samples", type=int, default=100, help="Number of samples to generate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: data/{occupation}_responses.jsonl)",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=4,
        help="Number of parallel processes (default: 4)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=10, help="Samples per batch (default: 10)"
    )
    parser.add_argument(
        "--no-validate", action="store_true", help="Skip validation step"
    )

    args = parser.parse_args()

    if not args.api_key:
        print(
            "Error: OpenAI API key required. Set OPENAI_API_KEY in .env file, environment variable, or use --api-key"
        )
        sys.exit(1)

    # Set default output path based on occupation
    if args.output is None:
        safe_occupation = args.occupation.lower().replace(" ", "_").replace("-", "_")
        args.output = f"data/{safe_occupation}_responses.jsonl"

    # Initialize generator
    generator = OccupationResponseGenerator(
        api_key=args.api_key,
        occupation=args.occupation,
        question_model=args.question_model,
        response_model=args.response_model,
        validator_model=args.validator_model,
        batch_size=args.batch_size,
    )

    # Generate dataset
    generator.generate_dataset(
        num_samples=args.num_samples,
        output_file=args.output,
        num_processes=args.processes,
        validate=not args.no_validate,
    )


if __name__ == "__main__":
    main()
