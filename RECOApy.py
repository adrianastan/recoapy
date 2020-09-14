from tkinter import Tk, Text, BOTH, W, N, E, S,NW, RAISED, font, PhotoImage
from tkinter import Button, Menu, DISABLED, NORMAL, LEFT, Label, StringVar, Canvas
from tkinter.ttk import Frame
from tkinter.filedialog import askopenfilename
import pyaudio
import os
from sys import platform
import hyperparams as hp
import threading
import numpy as np
import struct
import librosa
from datetime import datetime
from scipy.io.wavfile import write
from scipy import signal
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class RecApp(Frame):
    def __init__(self, root):
        super().__init__(padding=(15,15,15,15))
        self.root = root
        self.data = []
        self.safe_copy_counter  = 1
        self.timer = 0
        self.afterid = 0
        self.st = 0
        self.rec_init()
        self.initUI()

    ###########################
    ## RECORDING FUNCS
    ###########################
    def rec_init(self):
        self.chunk = 4096  # Record in chunks of 4096 samples
        self.sample_format = pyaudio.paInt16  # 16 bits per sample
        self.channels = 1
        self.fs = 48000  # Record at 48000 samples per second
        self.max_seconds = hp.max_seconds    
        self.filename_id = hp.filename_id
        self.output_folder = hp.output_folder
        self.prompt_index = hp.counter
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        self.wav_folder = os.path.join(self.output_folder, 'recs')
        if not os.path.exists(self.wav_folder):
            os.makedirs(self.wav_folder)

        self.safecopy_folder = os.path.join(self.output_folder, 'safe_copy')
        if not os.path.exists(self.safecopy_folder):
            os.makedirs(self.safecopy_folder)


        if hp.plot_specs:
            self.spec_folder = os.path.join(self.output_folder,'spectrograms')
            if not os.path.exists(self.spec_folder):
                os.makedirs(self.spec_folder)

        if hp.normalize_wavs:
            self.norm_folder = os.path.join(self.output_folder,'wav_norm')
            if not os.path.exists(self.norm_folder):
                os.makedirs(self.norm_folder)

        if hp.trim_wavs:
            self.trim_folder = os.path.join(self.output_folder,'wav_trim')
            if not os.path.exists(self.trim_folder):
                os.makedirs(self.trim_folder)

        self.init_log_file()
        self.session_start_time = datetime.now()
        self.p = pyaudio.PyAudio()  # Create an interface to PortAudio

    def init_log_file(self):
        # logging file
        log_file = os.path.join(self.output_folder, 'rec_log'+str(datetime.now())[:-7].replace(' ','_')+'.txt')
        self.log_file = open(log_file, 'w')

        self.log_file.write("Sampling frequency: "+str(self.fs)+"\n")
        self.log_file.write("Output folder: "+self.output_folder+"\n")
        self.log_file.write("Max seconds: "+str(self.max_seconds)+"\n")
        self.log_file.write("File id: "+str(self.filename_id)+"\n")
        self.log_file.write("Counter start: "+str(self.prompt_index)+"\n")
        

    def thread_record(self):
        self.timer=0
        self.run_timer()
        event = threading.Event()
        self.thread = threading.Thread(target=self.record, args=(event, ))
        #self.thread.setDaemon(True)
        self.thread.start()

    def record(self, event):
        start_time = datetime.now()
        self.rec_lbl['background'] = 'red'
        self.rec_lbl['text'] = 'Recording...'
        self.rec['state'] = DISABLED
        self.next['state'] = DISABLED
        self.prev['state'] = DISABLED
        self.stop_rec['state'] = NORMAL
        
        self.fig.clear()

        max_value = 0.0
        stream = self.p.open(format=self.sample_format,
                        channels=self.channels,
                        rate=self.fs,
                        frames_per_buffer=self.chunk,
                        input=True)
      

        # Store data in chunks for max_length seconds
        frames = []  # Initialize array to store frames
        i = 0
        speech = True
        self.st = 1
        while (i < int(self.fs / self.chunk * self.max_seconds)) and (self.st == 1):
            data = stream.read(self.chunk)
            frames.append(data)
            max_value = self.get_max_level(data, max_value)
            if max_value > 0.9:
                self.vu["fg"]= 'red'
            else:
                self.vu["fg"]= 'green'
            self.vu_level.set("MAX Amp: "+'{:.2f}'.format(max_value))
            if (i > 5 *self.fs / self.chunk) and int.from_bytes(data, byteorder='big')==0:
                self.rec_lbl['background'] = 'red'
                self.rec_lbl['text']= 'No speech detected for 5 seconds. Recording stopped!'
                speech = False
                break
            i+=1

        self.after_cancel(self.afterid)
        # Stop and close the stream 
        stream.stop_stream()
        stream.close()
        self.stop_rec['state'] = DISABLED

        formatted = "%dh"%(len(b''.join(frames))/2)
        self.audio_data = np.array([x/32768.0 for x in struct.unpack(formatted, b''.join(frames))])
        
        if np.max(self.audio_data) < 0.02:
            speech = False
            self.rec_lbl['text'] = 'No sound detected...'

        if speech:
            self.safe_copy['state'] = NORMAL
            self.rec['state'] = NORMAL
            self.nav_states()


            filename = self.filename_id+'_'+str(self.prompt_index).zfill(5)+'.wav'
            self.write_wave(os.path.join(self.wav_folder, filename), self.audio_data)
            duration = datetime.now() - start_time
            self.log('File saved to ' +os.path.join(self.wav_folder, filename)+" | Duration: "+str(duration)[2:-3])
            
            if hp.normalize_wavs:
                norm_audio = self.audio_data / np.linalg.norm(self.audio_data)               
                self.write_wave(os.path.join(self.norm_folder, filename), norm_audio)
                self.log('Norm file saved to ' +os.path.join(self.norm_folder, filename))
            
            if hp.trim_wavs:
                trim_audio, _ = librosa.effects.trim(self.audio_data)
                self.write_wave(os.path.join(self.trim_folder, filename), trim_audio)
                self.log('Trimmed file saved to ' +os.path.join(self.wav_folder, filename))



            self.plots()
            self.rec_lbl['background'] = 'blue'
            self.rec_lbl['text'] = 'File saved to: '+self.output_folder+'/'+filename
        else:
            self.rec['state'] = NORMAL

    def stop_rec(self):
        self.st = 0

    def read_file(self):
        name = askopenfilename(initialdir=os.getcwd(),
                               filetypes =(("Text File", "*.txt"),("All Files","*.*")),
                               title = "Choose a  prompt file."
                                )
        try:
            with open(name,'r') as f:
                self.data = [x.strip().replace('|', '\n') for x in f.readlines()]
                self.lbl['text']= self.data[self.prompt_index-1]

            self.nav_states()
            self.rec['state'] = NORMAL
            self.log_file.write("PROMPT FILE: "+ name+"\n\n")
        except:
            self.lbl['text']= "No prompt file selected! Use the menu above to load a transcript file!"
    

    ###################
    ## TIMER
    ###################
    def format_timer(self):
        secs = self.timer//10
        msecs = self.timer - secs*10
        return str(secs)+' s : '+str(msecs).ljust(3,'0')+' ms'        
        

    def run_timer(self):
        self.now.set(self.format_timer())
        self.timer+=1
        # schedule timer to call myself after 100 mseconds
        self.afterid = self.after(100, self.run_timer)    
    
    ###################
    ## MAX LEVEL
    ###################
    def get_rms(self, block):
        format = "%dh"%(len(block)/2)
        shorts = struct.unpack(format, block)
        return '{:.2f}'.format(np.max(np.abs(shorts))*1.0/32768.0)


    def get_max_level(self, block, max_value):
        format = "%dh"%(len(block)/2)
        shorts = struct.unpack(format, block)
        a = np.max(np.abs(shorts))*1.0/32768.0
        return np.max([max_value,a])
        


    #############################
    ## SAVE WAVS - Safe and MAIN
    #############################
    def save_safe_copy(self):
        filename = self.filename_id+'_'+str(self.prompt_index).zfill(5)+'_safecopy_'+str(self.safe_copy_counter).zfill(3)+'.wav'
        self.log('Safe copy saved to: '+os.path.join(self.safecopy_folder, filename))
        self.write_wave(os.path.join(self.safecopy_folder,filename), self.audio_data)
        self.safe_copy_counter +=1
        self.rec_lbl['background'] = 'green'
        self.rec_lbl['text'] = 'Safe copy saved to: '+self.output_folder+'/'+filename
        self.safe_copy['state'] = DISABLED

    def write_wave(self, filename, audio_data):
        out_data = np.array([int(x*32768) for x in audio_data], dtype='int16')
        write(filename, self.fs, out_data)
        '''
        wf = wave.open(os.path.join(self.output_folder,filename), 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.sample_format))
        wf.setframerate(self.fs)
        
        wf.writeframes(b''.join(self.frames))
        wf.close()
        '''
        


    def plots(self):
        ax = self.fig.add_subplot(121)
        ax.set_xlim(0.0, float(len(self.audio_data))/self.fs)
        ax.set_ylim(-1,1)
        ax.set_xlabel("Time [s]")
        time_axis = np.arange(0, len(self.audio_data))*1.00/self.fs
        norm_audio = self.audio_data *1/np.max(np.abs(self.audio_data))
        ax.plot(time_axis, self.audio_data)

        try:
            ax = self.fig.add_subplot(122)
            f, t, spect = signal.spectrogram(self.audio_data, self.fs)
            ax.pcolormesh(t, f/1000, 10*np.log10(spect))
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Frequency [kHz]")
            self.spectrogram.draw()

            
            if hp.plot_specs:
                spec_filename = self.filename_id+'_'+str(self.prompt_index).zfill(5)+'.png'
                out_file = os.path.join(self.spec_folder,spec_filename)
                extent = ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
                self.fig.savefig(out_file, bbox_inches=extent.expanded(1.1, 1.2))
        except:
            pass        

    #############################
    ## PROMPT NAVIGATION
    #############################



    def next_prompt(self):
        self.prompt_index +=1
        self.log ("Prompt index: "+ str(self.prompt_index))
        if self.prompt_index <= len(self.data):
            self.lbl['text']= self.data[self.prompt_index-1]
            self.safe_copy_counter = 1
            self.rec_lbl['text'] = ''
            self.safe_copy['state'] = DISABLED
            self.rec['state'] = NORMAL
            self.nav_states()
        else:
            self.rec['state'] = DISABLED
            self.nav_states()
            self.safe_copy['state'] = DISABLED
            

    def prev_prompt(self):
        self.prompt_index -=1
        self.log ("Prompt index: "+ str(self.prompt_index))
        if self.prompt_index >= 0:
            self.lbl['text']= self.data[self.prompt_index-1]
            self.safe_copy_counter = 1
            self.rec_lbl['text'] = ''
            self.safe_copy['state'] = DISABLED
            self.rec['state'] = NORMAL
            self.nav_states()
        

    def nav_states(self):
        if self.prompt_index < len(self.data):
            self.next['state'] = NORMAL
        else:
            self.next['state'] = DISABLED
        if self.prompt_index-1 > 0:
            self.prev['state'] = NORMAL   
        else:
            self.prev['state'] = DISABLED




    ###########################
    ## INIT UI
    ###########################
    def initUI(self):

        self.master.title("RECOApp")
        self.pack(fill=BOTH, expand=True)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(3, pad=7)
        self.rowconfigure(3, weight=1)
        self.rowconfigure(7, pad=7)



        self.fig = Figure(figsize=(20,2),facecolor='gainsboro')
        self.fig.subplots_adjust(bottom=0.25, top=0.90, left=0.05, right=0.99)
        self.spectrogram = FigureCanvasTkAgg(self.fig, master=self)
        self.spectrogram.get_tk_widget().grid(row=8, column=0, columnspan=3)
        
        ## if on MacOS, show buttons as the menu might not work properly
        if platform =='darwin':
        #if 1:
            self.load = Button(self, text="Load prompt", font=("Verdana",14),  compound = LEFT, command = self.read_file)
            self.load.grid(row=0, column=0, padx=15)

            self.quit = Button(self, text="X", font=("Verdana",14, "bold"),  compound = LEFT, command = quit)
            self.quit.grid(row=0, column=1, padx=15)
        
        self.lbl = Label(self, text="Load a prompt text file using the top menu...", 
                        background="bisque", relief=RAISED, borderwidth=1, anchor="c", wraplength=1000)
        self.lbl.config(font=("Verdana", 40))
        self.lbl.grid(row=1, column=0, columnspan=3, rowspan=4,
            padx=15, pady=15, sticky=E+W+S+N)

        self.rec_lbl = Label(self, anchor="c")
        self.rec_lbl.config(font=("Verdana", 12), background='red', fg="white",relief=RAISED, borderwidth=1)
        self.rec_lbl.grid(row=7, column=1, padx=15, pady=15)

        ## REC button
        self.rec = Button(self, text="Record", font=("Verdana",14, "bold"),  compound = LEFT, 
                    command = self.thread_record, state=DISABLED)
        self.rec.grid(row=6, column=1, padx=15)


        ## STOP button
        self.stop_rec = Button(self, text="STOP", font=("Verdana",14, "bold"),  compound = LEFT, 
                        command = self.stop_rec, state=DISABLED, fg='red')
        self.stop_rec.grid(row=6, column=2, padx=15)
        
        

        ## safe copy button
        self.safe_copy = Button(self, text="Safe Copy", font=("Verdana",14, "bold"), fg="blue",  state=DISABLED, command=self.save_safe_copy)
        self.safe_copy.grid(row=6, column=0, padx=15)
        
        ## next button
        self.next = Button(self, text="Next >", font=("Verdana",14), command = self.next_prompt, state=DISABLED)
        self.next.grid(row=7, column=2, padx=15)

        ## prev button
        self.prev = Button(self, text="< Previous", font=("Verdana",14), command = self.prev_prompt, state=DISABLED)
        self.prev.grid(row=7, column=0, padx=15)

        ## timer
        self.now = StringVar()
        self.time = Label(self, font=('Verdana', 24), text="0 s : 000 ms")
        self.time.grid(row=5, column=1, padx=15)
        self.time["textvariable"] = self.now

        ## vu meter
        self.vu_level = StringVar()
        self.vu = Label(self, font=('Verdana', 16))
        self.vu.grid(row=5, column=0, padx=15)
        self.vu["textvariable"] = self.vu_level


    def _quit(self):
        self.p.terminate()
        self.log_file.write("\n**********************\n")
        self.log_file.write("Total time in session: "+str(datetime.now() - self.session_start_time)[:-7])
        self.log_file.close()
        self.root.destroy()
        self.root.quit()
        
    def log(self, message):
        time = datetime.now()
        formatted_message = "["+time.strftime("%Y/%b/%d/ %H:%M:%S")+"] INFO: "+message
        print (formatted_message)
        self.log_file.write(formatted_message+"\n")
    

'''
MAIN FUNCTION
'''
def main():
    root = Tk()
    root.geometry("1300x800+300+300")
    app = RecApp(root)
    menubar = Menu(root)  
    menubar.add_command(label="Load prompt file |", command=app.read_file)  
    menubar.add_command(label="QUIT", command=app._quit) 
  #  root.attributes('-fullscreen',True)
    root.iconphoto(True, PhotoImage(file="logo.png"))
    root.config(menu=menubar)  
    root.protocol("WM_DELETE_WINDOW", app._quit)
    root.mainloop()

if __name__ == '__main__':
    main()


