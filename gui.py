import tkinter as tk
from tkinter import Scrollbar, Text, Button, PhotoImage
from chatbot import chatbot_response

# Function to send user message and get bot response
def send():
    user_message = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", tk.END)
    if user_message != '':
        ChatLog.config(state=tk.NORMAL)
        ChatLog.insert(tk.END, "You: " + user_message + '\n\n')
        ChatLog.config(foreground="#1a1a1a", font=("Verdana", 12))

        bot_response = chatbot_response(user_message)
        ChatLog.insert(tk.END, "Bot: " + bot_response + '\n\n')
        ChatLog.config(state=tk.DISABLED)
        ChatLog.yview(tk.END)

# Main window setup
base = tk.Tk()
base.title("Enrollment Inquiry Chatbot")
base.geometry("500x600")
base.resizable(width=False, height=False)
base.configure(bg="#e6e6e6")

# Adding a header
header_frame = tk.Frame(base, bd=0, bg="#0073e6", height=60)
header_frame.pack(fill="x", pady=(0, 10))
header_label = tk.Label(header_frame, text="Trincomalee Campus Inquiry Chatbot", font=("Arial", 16, "bold"), bg="#0073e6", fg="white")
header_label.pack(pady=10)

# Adding a chatbot logo (optional)
# logo = PhotoImage(file="path/to/logo.png")  # Uncomment and add your logo path here
# logo_label = tk.Label(header_frame, image=logo, bg="#0073e6")
# logo_label.pack(side="left", padx=20)

# Chat log
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial", wrap="word", padx=10, pady=10)
ChatLog.config(state=tk.DISABLED)

# Scrollbar
scrollbar = Scrollbar(base, command=ChatLog.yview)
ChatLog['yscrollcommand'] = scrollbar.set

# Send button
SendButton = Button(base, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5, bd=0, bg="#0073e6", activebackground="#005bb5", fg='#ffffff', command=send)

# Entry box for user input
EntryBox = Text(base, bd=0, bg="white", width="29", height="5", font="Arial", padx=10, pady=10)

# Positioning components on screen
scrollbar.place(x=476, y=70, height=386)
ChatLog.place(x=6, y=70, height=386, width=470)
EntryBox.place(x=6, y=470, height=90, width=370)
SendButton.place(x=380, y=470, height=90)

# Running the GUI
base.mainloop()



