const mongoose = require('mongoose');

//Employee
const Schema = mongoose.Schema;
const EmployeeSchema = new Schema({
  user: {
    type: Schema.ObjectId,
    ref: 'User',
    required: true,
  },
  name: {
    type: String,
    required: true,
  },
  sex: {
    type: String,
    required: true,
  },
  local: {
    type: String,
    required: true,
  },
  RRN: {
    type: String,
    required: true,
  },
  phonenumber: {
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

module.exports = mongoose.model('Employee', EmployeeSchema);