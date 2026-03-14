import type { Message } from "../types";

interface Props {
  message: Message;
}

export function MessageBubble({ message }: Props) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end mb-4">
        <div className="bg-blue-600 text-white rounded-lg px-4 py-2 max-w-[80%]">
          {message.content}
        </div>
      </div>
    );
  }

  if (message.role === "error") {
    return (
      <div className="flex justify-start mb-4">
        <div className="bg-red-900/50 border border-red-700 text-red-200 rounded-lg px-4 py-2 max-w-[80%]">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start mb-4">
      <div className="bg-gray-800 text-gray-100 rounded-lg px-4 py-3 max-w-[80%]">
        <div className="whitespace-pre-wrap">{message.content}</div>
        {message.warnings && message.warnings.length > 0 && (
          <div className="mt-2 border-t border-gray-700 pt-2">
            {message.warnings.map((w, i) => (
              <div
                key={i}
                className={`text-xs mt-1 ${
                  w.severity === "error"
                    ? "text-red-400"
                    : "text-yellow-400"
                }`}
              >
                [{w.check}] {w.message}
              </div>
            ))}
          </div>
        )}
        {message.disclaimer && (
          <div className="mt-2 text-xs text-gray-500 italic border-t border-gray-700 pt-2">
            {message.disclaimer}
          </div>
        )}
      </div>
    </div>
  );
}
