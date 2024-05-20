const mongoose = require('mongoose');

//추후 여기를 고객으로 변경하면 될듯
const Schema = mongoose.Schema;
const NoteSchema = new Schema({
  user: {
    type: Schema.ObjectId,
    ref: 'User'
  },
  title: {
    type: String,
    required: true,
  },
  body: {
    type: String,
    required: true,
  },
  createdAt: {
    type: Date,
    default: Date.now()
  },
  updatedAt: {
    type: Date,
    default: Date.now()
  }
});

module.exports = mongoose.model('Note', NoteSchema);


