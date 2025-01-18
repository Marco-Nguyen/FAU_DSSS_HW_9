from typing import Final
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from transformers import GPT2LMHeadModel, GPT2Tokenizer

TOKEN: Final = '7817313040:AAGrmu2p5oaDLPGRVdDRyNIg2iu26tpPOsk'

BOT_USERNAME: Final = '@dsss_hw_9_bot'

# model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# tokenizer = AutoTokenizer.from_pretrained(model_name) 
# model = AutoModelForCausalLM.from_pretrained(model_name)
# model.eval()

model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model.eval()

# Command
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Hello! Thanks for chatting with me! \
                                    I am your AI Assistant. How can I help you?')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Please type your questions or concerns')

async def custom_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Custom command')

# Responses
def generate_response(text: str) -> str:
    # Encode input text and generate response
    prompt = f"Answer the following question:\nQ: {text}\nA:"
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs, 
        max_length=200,  # Limit response length
        num_return_sequences=1,
        no_repeat_ngram_size=2,  # Avoid repetition
        top_p=0.95,  # Use top-p (nucleus) sampling for more focused output
        top_k=50,  # Limit the number of potential next tokens
        temperature=0.7,  # Control randomness (lower is more focused)
        pad_token_id=tokenizer.eos_token_id  # Handle padding token
    )
    
    # Decode and return the output text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("A:")[1].strip()

def handle_reponse(text: str):
    processed: str = text.lower()
    
    if 'hello' in processed:
        return 'Hey there'
    
    if 'how are you' in processed:
        return 'I am fine! Thanks'  
    return generate_response(processed)

# def handle_reponse(text: str):
#     processed: str = text.lower()    
#     if 'hello' in processed:
#         return 'Hey there'
    
#     if 'how are you' in processed:
#         return 'I am fine! Thanks'  
#     return "Sorry I don't undertand what you said..."


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message_type: str = update.message.chat.type
    text: str = update.message.text

    print(f'User ({update.message.chat.id}) in {message_type}: "{text}"')

    if message_type == 'group':
        if BOT_USERNAME in text:
            new_text: str = text.replace(BOT_USERNAME, '').strip()
            response: str = handle_reponse(new_text)
        else:
            return
    else:
        response: str = handle_reponse(text)

    print('Bot: ', response)
    await update.message.reply_text(response)


async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f'Update {update} caused error {context.error}')


if __name__ == '__main__':
    print('Starting bot...')
    app = Application.builder().token(TOKEN).build()

    # Commands
    app.add_handler(CommandHandler('Start', start_command))
    app.add_handler(CommandHandler('Help', help_command))
    app.add_handler(CommandHandler('Custom', custom_command))

    # Messages
    app.add_handler(MessageHandler(filters.TEXT, handle_message))
    
    # Polls the bot
    print('Polling...')
    app.run_polling(poll_interval=3)

    # # Errors
    # app.add_error_handler(error)

    

