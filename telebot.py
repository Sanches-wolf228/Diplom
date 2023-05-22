import io
import numpy as np
from PIL import Image
from skimage.io import imread
from telegram import InlineQueryResultArticle, InputTextMessageContent
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import Updater, CommandHandler
from telegram.ext import MessageHandler, Filters, InlineQueryHandler

text_start = \
'''Hello, I am FindYourFaceBot

With my help, you can automatically find all the photos with you by uploading your photo and uploading the album in which you need to search for it, or by attaching a link to Google Drive.
For more information about me use /help'''

help_dict = {
    "How to upload a photo?" :
'''Just send photos to the bot or
Use /load command with a link to the photo on the Internet or
Use /load command with a link to Google Drive folder

Only photos where faces are recognized will be saved''',

    "What is the main functionality" :
'''/find_face n - select all photos, containing person number n
/find_gdrive n https://drive.google.com/drive/folders/... - the same, but on the Google Drive, it will take some time to run
/find_party n1 n2 n3 - select all photos, that contains persons n1, n2, n3 e.t.c simultaniously
''',

    "What is the additional functionality" :
'''/show n1 n2 n3 - select thumbnails of faces n1, n2, n3...
/compare n1, n2 - show similarity of faces n1 and n2
/group_by_face - find faces of persons, who occur at least twice
/set_threshold 0.4 - set the threshold for 2 faces to belong to the same person

Low quality images need bigger thresholds to distinguish different people, by default the threshold equals to 0.4
''',

    "Learn more & tests" :
'''Visit https://github.com/Sanches-wolf228/Diplom to see tests'''
}

def similarity(embed1, embed2):
    return 1 - np.linalg.norm(embed1 - embed2)

def merge(img1, img2):
    h = img1.size[0]
    w = 2 * h

    img = Image.new("RGB", (w, h))
    img.paste(img1.resize((h, h)))
    img.paste(img2.resize((h, h)), (h, 0))

    return img

def dfs(mat, res, v, num):
    if res[v] != 0:
        return False
    res[v] = num
    for i in range(len(res)):
        if mat[v][i]:
            dfs(mat, res, i, num)
    return True

def comps(mat):
    num = 1
    res = [0] * len(mat)
    for i in range(len(res)):
        if dfs(mat, res, i, num):
            num += 1
    return res

class telebot():
    def __init__(self, token, face_detector, shape_predictor, face_recognition_model, drive = None, threshold = 0.4):
        self.token = token
        self.face_count = 0
        self.img_count = 0
        self.embeddings = [0]
        self.face_to_index = {}
        self.face_detector = face_detector
        self.shape_predictor = shape_predictor
        self.face_recognition_model = face_recognition_model
        self.drive = drive
        self.threshold = threshold

        self.updater = Updater(token = self.token)
        self.dispatcher = self.updater.dispatcher

        self.dispatcher.add_handler(CommandHandler('start', self.start))
        self.dispatcher.add_handler(CommandHandler('help', self.help))

        self.dispatcher.add_handler(CommandHandler('show', self.show))
        self.dispatcher.add_handler(CommandHandler('load', self.load))
        self.dispatcher.add_handler(CommandHandler('compare', self.compare))
        self.dispatcher.add_handler(CommandHandler('find_face', self.find_face))
        self.dispatcher.add_handler(CommandHandler('find_party', self.find_party))
        self.dispatcher.add_handler(CommandHandler('find_gdrive', self.find_gdrive))
        self.dispatcher.add_handler(CommandHandler('group_by_face', self.group_by_face))
        self.dispatcher.add_handler(CommandHandler('set_threshold', self.set_threshold))

        self.dispatcher.add_handler(MessageHandler(Filters.photo, self.handle_photo))
        self.dispatcher.add_handler(MessageHandler(Filters.text, self.text_request))
    
    def start(self, update, context):
        self.face_count = 0
        self.img_count = 0
        self.embeddings = [0]
        self.face_to_index = {}
        
        context.bot.send_message(chat_id=update.message.chat_id, text=text_start)
      
    def help(self, update, context):
        reply_keyboard = [["What is the main functionality", "What is the additional functionality"],
                          ["How to upload a photo?", "Learn more & tests"],
                          ["Close help"]]

        update.message.reply_text("Choose option", reply_markup = ReplyKeyboardMarkup(reply_keyboard))

    def text_request(self, update, context):
        message = update.message
        text = message.text

        if text in help_dict.keys():
            context.bot.send_message(chat_id=update.message.chat_id, text=help_dict[text])
        elif text == "Close help":
            update.message.reply_text("Help is closed", reply_markup = ReplyKeyboardRemove())
        else:
            context.bot.send_message(chat_id=update.message.chat_id, text="Unknown command, try to use /help")
            

    def save_image(self, context, update, image, send = False):
        faces = self.face_detector(np.array(image), 1)
        if len(faces) == 0:
            if not send:
                context.bot.send_message(chat_id=update.message.chat_id, text='No faces found')
            return
        

        self.img_count += 1
        filename = "img"+str(self.img_count)+".jpg"
        image.save(filename)

        if send:
            context.bot.send_photo(chat_id=update.message.chat_id, photo = open(filename, 'rb'), caption = 'Found ' + str(len(faces)) + " face(s)")
        else:
            context.bot.send_message(chat_id=update.message.chat_id, text='Found ' + str(len(faces)) + " face(s)")

        for f in faces:
            landmarks = self.shape_predictor(np.array(image), f)
            embedding = self.face_recognition_model.compute_face_descriptor(np.array(image), landmarks)
            self.embeddings.append(np.array(embedding))

            self.face_count += 1
            filename = "face"+str(self.face_count)+".jpg"
            image.crop([f.left(), f.top(), f.right(), f.bottom()]).resize((256, 256)).save(filename)
            self.face_to_index[self.face_count] = self.img_count
            context.bot.send_photo(chat_id=update.message.chat_id, photo = open(filename, 'rb'), caption = "Face " + str(self.face_count))

    def show(self, update, context):
        message = update.message
        words = message.text.split()

        try:
            faces = list(map(int, words[1:]))
        except:
            context.bot.send_message(chat_id=update.message.chat_id, text="It seems, some of your arguments are not integer numbers")
            return
        
        if len(faces) == 0:
            return
        
        if (min(faces) < 1) or (max(faces) > self.face_count):
            context.bot.send_message(chat_id=update.message.chat_id, text="It seems, some of your arguments are out of range, the amount of photos is " + str(self.face_count))
            return
        
        for x in faces:
            filename = "face"+str(x)+".jpg"
            context.bot.send_photo(chat_id=update.message.chat_id, photo = open(filename, 'rb'), caption = "Face " + str(x))
    
    def load(self, update, context):
        message = update.message
        words = message.text.split()

        if len(words) != 2:
            context.bot.send_message(chat_id=update.message.chat_id, text="Your request should look like '/load https://...'")
            return
        
        link = words[1]

        if not link.startswith('https://drive.google.com/drive/folders/'):
            try:
                image = Image.fromarray(imread(link))
            except BaseException as ex:
                context.bot.send_message(chat_id=update.message.chat_id, text= str(ex) + "\nCheck if the file is able to open by the link")
                return
            self.save_image(context, update, image)
            return
        
        file_list = self.drive.ListFile({'q': "'" + link[39:] + "' in parents"}).GetList()

        images = 0
        for file in file_list:
            file.GetContentFile("local")
            try:
                image = imread("local")
            except:
                # in case there are files, not including images
                continue
            
            self.save_image(context, update, Image.fromarray(image), send = True)
            images += 1
        context.bot.send_message(chat_id=update.message.chat_id, text="Loading is done, " + str(images) + " images found")

    def handle_photo(self, update, context):
        message = update.message
        photo = message.photo[~0]
        with io.BytesIO() as fd:
            file_id = context.bot.get_file(photo.file_id)
            file_id.download(out=fd)
            fd.seek(0)

            image = Image.open(fd)
            image.load()
        
        self.save_image(context, update, image)
        
    def compare(self, update, context):
        message = update.message
        words = message.text.split()

        if len(words) != 3:
            context.bot.send_message(chat_id=update.message.chat_id, text="Your request should look like '/compare x y'")
            return
        
        try:
            n1, n2 = int(words[1]), int(words[2])
        except:
            context.bot.send_message(chat_id=update.message.chat_id, text="It seems, your arguments are not integer numbers")
            return
        
        if (n1 < 1) or (n2 < 1) or (n1 > self.face_count) or (n2 > self.face_count):
            context.bot.send_message(chat_id=update.message.chat_id, text="It seems, your arguments are out of range, the amount of photos is " + str(self.face_count))
        else:
            sim = similarity(self.embeddings[n1], self.embeddings[n2])
            img1 = Image.open("face"+str(n1)+".jpg")
            img2 = Image.open("face"+str(n2)+".jpg")
            merge(img1, img2).save("comparison.jpg")
            context.bot.send_photo(chat_id=update.message.chat_id, photo = open("comparison.jpg", 'rb'),
                                  caption = "Similarity of this photos is " + str(round(sim, 4)))


    def find_face(self, update, context):
        message = update.message
        words = message.text.split()

        if len(words) != 2:
            context.bot.send_message(chat_id=update.message.chat_id, text="Your request should look like '/find_face n'")
            return
        
        try:
            n = int(words[1])
        except:
            context.bot.send_message(chat_id=update.message.chat_id, text="It seems, your argument is not an integer number")
            return
        
        if (n < 1) or (n > self.face_count):
            context.bot.send_message(chat_id=update.message.chat_id, text="It seems, your argument is out of range, the amount of photos is " + str(self.face_count))
            return
        
        photos = set([self.face_to_index[i] for i in range(1, self.face_count + 1) if similarity(self.embeddings[i], self.embeddings[n]) > self.threshold])

        context.bot.send_message(chat_id=update.message.chat_id, text="Found " + str(len(photos)) + " photo(s)")
        for num in photos:
            context.bot.send_photo(chat_id=update.message.chat_id, photo = open("img" + str(num) + ".jpg", 'rb'))

    def find_party(self, update, context):
        message = update.message
        words = message.text.split()
        
        try:
            persons = list(map(int, words[1:]))
        except:
            context.bot.send_message(chat_id=update.message.chat_id, text="It seems, some of your arguments are not integer numbers")
            return
        
        if (min(persons) < 1) or (max(persons) > self.face_count):
            context.bot.send_message(chat_id=update.message.chat_id, text="It seems, some of your arguments are out of range, the amount of photos is " + str(self.face_count))
            return

        persons = set(persons)
        
        photos = set.intersection(
            *[set([self.face_to_index[i] for i in range(1, self.face_count + 1) if similarity(self.embeddings[i], self.embeddings[person]) > self.threshold]) for person in persons]
        )

        context.bot.send_message(chat_id=update.message.chat_id, text="Found " + str(len(photos)) + " photo(s)")
        for num in photos:
            context.bot.send_photo(chat_id=update.message.chat_id, photo = open("img" + str(num) + ".jpg", 'rb'))

    def find_gdrive(self, update, context):
        message = update.message
        words = message.text.split()

        if len(words) != 3:
            context.bot.send_message(chat_id=update.message.chat_id, text="Your request should look like '/find_gdrive 4 https://drive.google.com/drive/folders/...'")
            return
        
        try:
            n = int(words[1])
        except:
            context.bot.send_message(chat_id=update.message.chat_id, text="It seems, your argument is not an integer number")
            return

        link = words[2]
        if not link.startswith('https://drive.google.com/drive/folders/'):
            context.bot.send_message(chat_id=update.message.chat_id, text="The link to Google Drive folder should start with 'https://drive.google.com/drive/folders'")
            return
        
        if (n < 1) or (n > self.face_count):
            context.bot.send_message(chat_id=update.message.chat_id, text="It seems, your argument is out of range, the amount of photos is " + str(self.face_count))
            return
        
        emb = self.embeddings[n]

        file_list = self.drive.ListFile({'q': "'" + link[39:] + "' in parents"}).GetList()

        images = 0

        for file in file_list:
            file.GetContentFile("local")
            # print(file["title"])
            try:
                image = imread("local")
            except:
                # in case there are files, not including images
                continue

            faces = self.face_detector(image, 1)

            for f in faces:
                landmarks = self.shape_predictor(image, f)
                new_emb = self.face_recognition_model.compute_face_descriptor(image, landmarks)
                if similarity(emb, new_emb) > self.threshold:
                    context.bot.send_photo(chat_id=update.message.chat_id, photo = open("local", 'rb'))
                    images += 1
                    break
            else:
                continue
        context.bot.send_message(chat_id=update.message.chat_id, text="Searching is done, " + str(images) + " face(s) found")

    def group_by_face(self, update, context):
        if self.face_count == 0:
            context.bot.send_message(chat_id=update.message.chat_id, text="No faces is downloaded")
            return
        
        distances = [
            [similarity(self.embeddings[i], self.embeddings[j]) > self.threshold for i in range(1, self.face_count + 1)] for j in range(1, self.face_count + 1)
        ]
        indexes = comps(distances)

        if max(indexes) == self.face_count:
            context.bot.send_message(chat_id=update.message.chat_id, text="No repeated faces found")
            return
        
        result = ""

        group = 1
        for g in range(1, max(indexes) + 1):
            faces = [str(i + 1) for i in range(self.face_count) if indexes[i] == g]
            if len(faces) > 1:
                result += "Group " + str(group) + ": `" + " ".join(faces) + "`\n"
                group += 1
        
        context.bot.send_message(chat_id=update.message.chat_id, text=result, parse_mode = 'markdown')
    
    def set_threshold(self, update, context):
        message = update.message
        words = message.text.split()

        if len(words) != 2:
            context.bot.send_message(chat_id=update.message.chat_id, text="Your request should look like '/set_threshold p'")
            return
        
        try:
            p = float(words[1])
        except:
            context.bot.send_message(chat_id=update.message.chat_id, text="It seems, your argument is not a float number")
            return
        
        if (p < 0) or (p > 1):
            context.bot.send_message(chat_id=update.message.chat_id, text="It seems, your argument is out of interval [0; 1]")
            return
        
        self.threshold = p
        context.bot.send_message(chat_id=update.message.chat_id, text="The threshold is now " + str(p))

    def run(self):
        self.updater.start_polling()
        self.updater.idle()