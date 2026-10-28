from rest_framework import viewsets
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import action
from rest_framework.response import Response
import requests

from .models import ChatSession, ChatMessage
from .serializers import ChatSessionSerializer, ChatMessageSerializer


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:latest"


class ChatSessionViewSet(viewsets.ModelViewSet):
    serializer_class = ChatSessionSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return ChatSession.objects.filter(
            user=self.request.user
        ).prefetch_related("messages")

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class ChatMessageViewSet(viewsets.ModelViewSet):
    serializer_class = ChatMessageSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return ChatMessage.objects.filter(
            session__user=self.request.user
        ).order_by("created_at")

    @action(detail=False, methods=["post"], url_path="send")
    def send_message(self, request):
        session_id = request.data.get("session")
        user_message = request.data.get("content")

        if not session_id:
            return Response(
                {"detail": "Session ID is required."},
                status=400
            )

        if not isinstance(user_message, str) or not user_message.strip():
            return Response(
                {"detail": "Message content is required."},
                status=400
            )

        try:
            session = ChatSession.objects.get(
                id=session_id,
                user=request.user
            )
        except ChatSession.DoesNotExist:
            return Response(
                {"detail": "Chat session not found."},
                status=404
            )

        user_message = user_message.strip()

        # Build the conversation before saving the latest user message.
        previous_messages = ChatMessage.objects.filter(
            session=session
        ).order_by("created_at", "id")

        conversation_lines = []

        for message in previous_messages:
            if message.role == "user":
                conversation_lines.append(f"User: {message.content}")
            elif message.role == "assistant":
                conversation_lines.append(
                    f"MediGuide AI: {message.content}"
                )

        conversation_lines.append(f"User: {user_message}")
        conversation = "\n".join(conversation_lines)

        prompt = f"""
You are MediGuide AI, an educational medical information assistant.

Rules:
- Provide general educational health information, not a diagnosis.
- Do not prescribe medicines or provide medication dosages.
- Explain medical concepts in simple language.
- Do not claim certainty about a person's medical condition.
- For potentially urgent symptoms, advise the user to seek prompt
  professional medical care or emergency assistance as appropriate.
- Encourage users to consult qualified healthcare professionals
  for personal medical decisions.
- Be clear, helpful, and concise.

Conversation:
{conversation}

Respond to the latest user message.
"""

        # Save the user message only after preparing the conversation.
        user_chat = ChatMessage.objects.create(
            session=session,
            role="user",
            content=user_message
        )

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=180
            )

            response.raise_for_status()
            data = response.json()

            ai_response = data.get("response", "").strip()

            if not ai_response:
                raise ValueError("Ollama returned an empty response.")

            assistant_message = ChatMessage.objects.create(
                session=session,
                role="assistant",
                content=ai_response
            )

            return Response({
                "session_id": session.id,
                "user_message": user_chat.content,
                "ai_response": ai_response,
                "message_id": assistant_message.id
            })

        except requests.RequestException as error:
            return Response(
                {
                    "detail": "MediGuide AI could not connect to Ollama.",
                    "error": str(error)
                },
                status=503
            )

        except (ValueError, KeyError) as error:
            return Response(
                {
                    "detail": "MediGuide AI received an invalid response.",
                    "error": str(error)
                },
                status=502
            )
