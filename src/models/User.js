const mongoose = require('mongoose');

const Schema = mongoose.Schema;
const UserSchema = new Schema({
  googleId: {
    type: String,
    required: true
  },
  displayName: {
    type: String,
    required: true
  },
  firstName: {
    type: String,
    required: true
  },
  lastName: {
    type: String,
    required: true
  },
  //발신번호 설정 위해 가입자(사용자) 번호 필요
  userphonenumber: {
    type: String,
    required: false
  },
  profileImage: {
    type: String,
    required: true
  },
  createdAt: {
    type: Date,
    default: Date.now
  }
});

module.exports = mongoose.model('User', UserSchema);