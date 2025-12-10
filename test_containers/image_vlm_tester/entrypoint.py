import argparse
import json
import sys

import httpx


class ImageVLMTester:
    def __init__(self, systems_params: dict, test_params: dict):
        self.systems_params = systems_params
        self.test_params = test_params

        self.image_gen_system_params = self.systems_params.get("system_under_test", {})
        self.vlm_system_params = self.systems_params.get("vlm_system", {})

        if not self.image_gen_system_params:
            raise ValueError("Missing system_under_test in systems_params")
        if not self.vlm_system_params:
            raise ValueError("Missing vlm_system in systems_params")

        self.image_gen_base_url = self.image_gen_system_params.get("base_url")
        self.image_gen_model = self.image_gen_system_params.get("model")
        self.image_gen_api_key = self.image_gen_system_params.get("api_key")

        self.vlm_base_url = self.vlm_system_params.get("base_url")
        self.vlm_model = self.vlm_system_params.get("model")
        self.vlm_api_key = self.vlm_system_params.get("api_key")

        if not all(
            [self.image_gen_base_url, self.image_gen_model, self.image_gen_api_key]
        ):
            raise ValueError("Incomplete image generation system configuration")
        if not all([self.vlm_base_url, self.vlm_model, self.vlm_api_key]):
            raise ValueError("Incomplete VLM system configuration")

        self.prompt = self.test_params.get("prompt")
        self.score_instruction = self.test_params.get(
            "score_instruction",
            "Please evaluate the following image for its aesthetic qualities and provide a score between 1 and 10.",
        )

        if not self.prompt:
            raise ValueError("Missing prompt in test_params")

    async def generate_image(self) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.image_gen_api_key}",
        }
        data = {
            "model": self.image_gen_model.split("/")[
                -1
            ],  # litellm model names usually have a prefix like 'openai/', so we need to remove it
            "prompt": self.prompt,
            "n": 1,
            "size": "1024x1024",
            "response_format": "url",
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.image_gen_base_url}/images/generations",
                headers=headers,
                json=data,
                timeout=60.0,
            )
            response.raise_for_status()
            response_json = response.json()
            return response_json["data"][0]["url"]

    async def score_image_with_vlm(self, image_url: str) -> float:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.vlm_api_key}",
        }

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.score_instruction},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]

        data = {
            "model": self.vlm_model.split("/")[
                -1
            ],  # litellm model names usually have a prefix like 'openai/', so we need to remove it
            "messages": messages,
            "max_tokens": 150,
            "temperature": 0.5,
        }

        print(f"DEBUG: VLM API Request Data: {json.dumps(data)}", file=sys.stderr)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.vlm_base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=60.0,
            )
            response.raise_for_status()
            response_json = response.json()
            content = response_json["choices"][0]["message"]["content"]
            # Extract score from content, e.g., "Score: 8/10" or similar
            try:
                # Attempt direct conversion to float, as VLM is instructed to return just the number.
                return float(content)
            except ValueError:
                # Fallback to regex if direct conversion fails (e.g., if VLM output format changes).
                score_match = re.search(r"\\b([0-9]|10)\b", content)
                if score_match:
                    return float(score_match.group(0))
                else:
                    print(
                        f"Warning: Could not extract score from VLM response: {content}",
                        file=sys.stderr,
                    )
                    return 0.0
            except Exception as e:
                print(f"Error parsing VLM response for score: {e}", file=sys.stderr)
                return 0.0

    async def run_test(self):
        test_result = {"success": False, "aesthetic_score": 0.0, "image_url": None}
        try:
            image_url = await self.generate_image()
            test_result["image_url"] = image_url
            aesthetic_score = await self.score_image_with_vlm(image_url)
            test_result["aesthetic_score"] = aesthetic_score
            test_result["success"] = True
        except Exception as e:
            test_result["error"] = str(e)
            print(f"Error during test execution: {e}", file=sys.stderr)
        finally:
            print(json.dumps(test_result, indent=2))
            sys.exit(0 if test_result["success"] else 1)


async def main():
    parser = argparse.ArgumentParser(description="Image VLM Tester")
    parser.add_argument(
        "--systems-params", required=True, help="Systems parameters as JSON string"
    )
    parser.add_argument(
        "--test-params", required=True, help="Test parameters as JSON string"
    )
    args = parser.parse_args()

    try:
        systems_params = json.loads(args.systems_params)
        test_params = json.loads(args.test_params)
        tester = ImageVLMTester(systems_params, test_params)
        await tester.run_test()
    except json.JSONDecodeError as e:
        error_result = {"success": False, "error": f"Invalid JSON in arguments: {e}"}
        print(json.dumps(error_result, indent=2))
        sys.exit(1)
    except ValueError as e:
        error_result = {"success": False, "error": str(e)}
        print(json.dumps(error_result, indent=2))
        sys.exit(1)
    except Exception as e:
        error_result = {"success": False, "error": f"Unexpected error: {e}"}
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    import asyncio
    import re

    asyncio.run(main())
