#import
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import os

#set up window
window = tk.Tk()
window.attributes('-fullscreen', True)
window.title('Enter ID')
window.resizable(0, 0)

#set up rows
window.rowconfigure(0,weight=1)
window.rowconfigure(1,weight=1)

ID_var = tk.StringVar()
session_var = tk.StringVar()

def nextPage():
	ID = str(ID_var.get())
	sesh = str(session_var.get())
	folder = ID + "_" + sesh
	if sesh == '' or ID == '':
		pass
	else:
		create_subfolder()
		with open(r"D:\GUI_data\temp.txt", 'w') as f:
			f.write(folder)
		window.destroy()
		# import consent
		
def quitPage():
	window.destroy()
		
def create_subfolder():
	os.remove(r"D:\GUI_data\temp.txt")
	ID = str(ID_var.get())
	sesh = str(session_var.get())
	folder = ID + "_" + sesh
	path = os.path.join(r"D:\New Data PEACE Experiment", folder)
	os.makedirs(path)

#ID Label
PageLabel = ttk.Label(window, text = "Please Enter Your ID and Session Number", font=("Arial", 25), justify='center', anchor='center')
PageLabel.place(anchor = 'center', relx = .5, rely = .2)

#ID Entry
IDEntry = ttk.Entry(window, textvariable = ID_var, width = 20, font = 10)
IDEntry.place(anchor = 'center', relx = .5, rely = .3)

IDLabel = ttk.Label(window, text = "ID: ", font = 15)
IDLabel.place(anchor = 'center', relx = .41, rely = .3)

#Session Entry
SessionEntry = ttk.Entry(window, textvariable = session_var, width = 20, font = 10)
SessionEntry.place(anchor = 'center', relx = .5, rely = .35)

SessionLabel = ttk.Label(window, text = "Session: ", font = 15)
SessionLabel.place(anchor = 'center', relx = .4, rely = .35)

#next button
nextButton = ttk.Button(window, text = "Next", command = nextPage)
nextButton.place(anchor = 'center', relx = .7, rely = .8)

#quit
quitButton = ttk.Button(window, text = "Quit", command = quitPage)
quitButton.place(anchor = 'center', relx = .3, rely = .8)

window.mainloop()